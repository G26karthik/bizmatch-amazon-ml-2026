"""Add unassigned one-token records (initials / web handle / pseudo-brand) to the only S1 at a format-tolerant equal address (d32)
(house number + street tokens, city/region words dropped; scripts/deep/d23_one_token_rule.py). Train truth with every
S1 visible (test has no hidden-S1 copies): initials 0.997-1.0, handles 1.0, pseudo-brands 0.992-0.993 precision.
Pseudo-brands at addresses with 2+ S1s and ambiguous initials are skipped; so are pairs the model scored and rejected
outside the uncertain zone (p <= 0.02). Kept: never-candidate pairs and uncertain-zone best pairs. Caps enforced.

uv run --no-project --with polars==1.44.2 python scripts/one_token_add.py <base> <out>
"""
import sys
from pathlib import Path

import polars as pl

KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["initials", "handle", "pseudo"]
SP = Path("local_data/work")
base, out = Path(sys.argv[1]), Path(sys.argv[2])
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e for ms in members.values() for e in ms}
t = pl.read_parquet(SP / "one_token_fuzzy_test.parquet").filter(pl.col("kind").is_in(KINDS))
from collections import defaultdict
cp = pl.read_csv(base / "candidate_pairs.tsv", separator="	", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = defaultdict(set)
for s_, m_ in zip(cp["source1_entity_id"].to_list(), cp["candidate_entity_ids"].fill_null("").to_list()):
    for q in (m_.split(",") if m_ else []):
        cand[q].add(s_)
rid = pl.concat([pl.read_parquet(f"local_data/test_source{x}.parquet", columns=["entity_id"]) for x in (1, 2, 3)]).with_row_index("rid")
rid_of = dict(zip(rid["entity_id"].to_list(), rid["rid"].to_list()))
unc = pl.read_parquet("local_data/llm/unc_test.parquet")
ubest = dict(zip(unc["q"].to_list(), unc["s1"].to_list()))
n, cap, rej = {}, 0, 0
for m, s, c, k in t.select("m", "s1", "country", "kind").iter_rows():
    if m in assigned or s not in members:
        continue
    if s in cand.get(m, ()) and ubest.get(rid_of[m]) != rid_of[s]:
        rej += 1; continue
    if sum(1 for x in members[s] if x[:2] == m[:2]) >= (5 if m[:2] == "S2" else 6):
        cap += 1; continue
    members[s].add(m); assigned.add(m); n[(c, k)] = n.get((c, k), 0) + 1
print("added:", dict(sorted(n.items())), "total", sum(n.values()), "skipped at cap", cap, "skipped (model-rejected candidates)", rej)
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order:
        f.write(f"{s}\t{','.join(sorted(members[s]))}\n")
c = pl.read_csv(base / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
        have = m.split(",") if m else []
        f.write(f"{s}\t{','.join(have + sorted(members[s] - set(have)))}\n")
print("wrote", out)
