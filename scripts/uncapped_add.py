"""frXX = base + US/India uncapped-retrieval adds (test-like scorer: TESTLIKE=1 unc_train.py apply) (unc_train.py apply -> local_data/work/uncapped_adds.parquet: empty-address
records never matched whose unique-best S1 by uncapped name key is outside their candidate list; train-truth OOF
precision 0.948, expected +65 S1-units). France adds are left out (not validated). Caps and one S1 per record enforced.

uv run --no-project --with polars==1.44.2 python scripts/uncapped_add.py <base> <out>
"""
from pathlib import Path

import polars as pl

SP = Path("local_data/work")
import sys
base, out = Path(sys.argv[1]), Path(sys.argv[2])
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e for ms in members.values() for e in ms}
add = pl.read_parquet(SP / "uncapped_adds.parquet").filter(pl.col("country") != "France")
n = 0
for e, s in add.select("m", "s1").iter_rows():
    if e in assigned or s not in members:
        continue
    src = e[:2]
    if sum(1 for x in members[s] if x[:2] == src) >= (5 if src == "S2" else 6):
        continue
    members[s].add(e); assigned.add(e); n += 1
print(f"added {n} of {add.height} US/India uncapped pairs")
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
        extra = sorted(members[s] - set(have))
        f.write(f"{s}\t{','.join(have + extra)}\n")
print("wrote", out)
