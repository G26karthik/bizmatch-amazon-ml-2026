"""Apply a list of proposed (record -> S1) adds to a submission: only currently unassigned records, caps (5 S2 / 6 S3)
and one S1 per record enforced; pairs the model scored and rejected (the pair is a candidate but not the record's
uncertain-zone best pair, i.e. production p <= 0.02 or another S1 preferred) are skipped unless --keep_rejected.
The candidate file gets every new matched pair appended.

uv run --no-project --with polars==1.44.2 python scripts/generic_add.py <base_dir> <out_dir> <adds.parquet> [--keep_rejected]
adds.parquet columns: m (record entity_id), s1 (S1 entity_id), country, kind
"""
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

base, out, adds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keep_rejected = "--keep_rejected" in sys.argv
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e for ms in members.values() for e in ms}
c = pl.read_csv(base / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = defaultdict(set)
for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        cand[q].add(s)
rid = pl.concat([pl.read_parquet(f"local_data/test_source{x}.parquet", columns=["entity_id"]) for x in (1, 2, 3)]).with_row_index("rid")
rid_of = dict(zip(rid["entity_id"].to_list(), rid["rid"].to_list()))
unc = pl.read_parquet("local_data/llm/unc_test.parquet")
ubest = dict(zip(unc["q"].to_list(), unc["s1"].to_list()))
t = pl.read_parquet(adds)
if "kind" not in t.columns:
    t = t.with_columns(pl.lit("add").alias("kind"))
n, cap, rej, skip = defaultdict(int), 0, 0, 0
for m, s, ct, k in t.select("m", "s1", "country", "kind").iter_rows():
    if m in assigned or s not in members:
        skip += 1; continue
    if not keep_rejected and s in cand.get(m, ()) and ubest.get(rid_of[m]) != rid_of[s]:
        rej += 1; continue
    if sum(1 for x in members[s] if x[:2] == m[:2]) >= (5 if m[:2] == "S2" else 6):
        cap += 1; continue
    members[s].add(m); assigned.add(m); n[(ct, k)] += 1
print("added:", dict(sorted(n.items())), "total", sum(n.values()), "| skipped: already assigned/unknown", skip, "model-rejected", rej, "cap", cap)
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order:
        f.write(f"{s}\t{','.join(sorted(members[s]))}\n")
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
        have = m.split(",") if m else []
        f.write(f"{s}\t{','.join(have + sorted(members[s] - set(have)))}\n")
print("wrote", out)
