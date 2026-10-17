"""fr15 = fr14 + France pseudo-brand / initials copies at a single S1's exact address (blind audit A:
blindA_fr_pseudo_initials_candidates.tsv; US truth 99.5% / 100% copies; full-cap test p~0.01) - leftover sibling-decoy
vocabulary swaps (blind audit B: blindB_rule_vocab_residual.tsv; flat at the cap, ~0.9 precision). Adds only unassigned
records, respects the 5 S2 / 6 S3 caps against the updated set, keeps row order; candidate_pairs = fr14's + new matches.

uv run --no-project --with polars==1.44.2 python scripts/fr15_build.py
"""
from pathlib import Path

import polars as pl

SP = Path("local_data/work")
base, out = Path("output/v28q_fr14"), Path("output/v28q_fr15")
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e: s for s, ms in members.items() for e in ms}
rm = pl.read_csv(SP / "nb/blindB_rule_vocab_residual.tsv", separator="\t")
n_rm = 0
for q, s in rm.select("q", "s1").iter_rows():
    if q in members.get(s, ()):
        members[s].discard(q); assigned.pop(q, None); n_rm += 1
add = pl.read_csv(SP / "blindA_fr_pseudo_initials_candidates.tsv", separator="\t").filter(pl.col("below_cap"))
n_add, skip = 0, 0
for e, s in add.select("entity_id", "s1").iter_rows():
    if e in assigned or s not in members:
        skip += 1; continue
    src = e[:2]
    if sum(1 for x in members[s] if x[:2] == src) >= (5 if src == "S2" else 6):
        skip += 1; continue
    members[s].add(e); assigned[e] = s; n_add += 1
print(f"removed {n_rm} of {rm.height} vocab-swap pairs; added {n_add} of {add.height} pseudo/initials pairs (skipped {skip})")
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order:
        f.write(f"{s}\t{','.join(sorted(members[s]))}\n")
c = pl.read_csv(base / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = {s: (m.split(",") if m else []) for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list())}
extra = 0
for s in order:
    have = set(cand[s])
    for e in sorted(members[s] - have):
        cand[s].append(e); extra += 1
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s in c["source1_entity_id"].to_list():
        f.write(f"{s}\t{','.join(cand[s])}\n")
print("candidates added for new matches:", extra, "->", out)
