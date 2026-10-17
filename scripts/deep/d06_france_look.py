"""Deep dive 6: raw look at French S1s: accepted records in fr16 + uncertain candidates (unc_test, production prob).

uv run --no-project --with polars==1.44.2 python scripts/deep/d06_france_look.py [n] [seed]
"""
import sys

import polars as pl

n, seed = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (12, 1)
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid")
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list()
rid_of = {e: i for i, e in enumerate(eid)}
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = {r["source1_entity_id"]: [e for e in (r["matched_entity_ids"] or "").split(",") if e] for r in g.iter_rows(named=True)}
assigned = {e: s for s, l in acc.items() for e in l}
unc = pl.read_parquet("local_data/llm/unc_test.parquet")
fr = rec.filter((pl.col("country") == "France") & pl.col("entity_id").str.starts_with("S1")).sample(n, seed=seed)
for s in fr["entity_id"].to_list():
    r = rid_of[s]
    print(f"\n=== {s} | {nm[r]!r} | {ad[r]}")
    for e in acc.get(s, []):
        print(f"   ACC  {e:14s} {nm[rid_of[e]]!r:50s} | {ad[rid_of[e]]}")
    for q, p in unc.filter(pl.col("s1") == r).select("q", "prob").iter_rows():
        if eid[q] in acc.get(s, []):
            continue
        where = assigned.get(eid[q], "-")
        print(f"   unc  {eid[q]:14s} p={p:.3f} {nm[q]!r:44s} | {ad[q]}  [assigned to {where}]")
