"""Removal candidates: France accepted pairs whose record is a same-address one-word swap to a type word (not a generic
copy word) AND whose legal form differs from the S1's (changed, dropped or added). Mixture estimates put ~94% of these
at decoys (France copies keep the legal form 97.6%; decoys 51%; decoy share among same-address type swaps ~0.42).
Reports counts and the expected F gain per S1 for a range of decoy precisions, using each S1's accepted count.

python scripts/sibling_legal_removal.py /work/out_fr7
"""
import sys

import polars as pl

out = sys.argv[1]
sw = pl.read_parquet("/work/dive/fr_swaps_feat.parquet")
cand = sw.filter((pl.col("cat") == "same|TYPE") & pl.col("acc") & (pl.col("legal_rel") != "same")).select("q", "s1", "w_in", "legal_rel").unique(["q", "s1"])
print("candidates:", cand.height, cand.group_by("legal_rel").len().rows())
print("top words:", cand.group_by("w_in").len().sort("len", descending=True).head(15).rows())
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(rec.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(rec.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
k = acc.group_by("s1").agg(pl.len().alias("k"))
per = cand.group_by("s1").agg(pl.len().alias("r")).join(k, on="s1")


def f05(tp, fp, fn):
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 1.25 * p * r / (0.25 * p + r)


# per S1 with r removals among k accepted: assume other k-r are true and all true copies found;
# each removed record is a decoy with prob pi (independently)
from math import comb

for pi in (0.5, 0.7, 0.85, 0.94):
    tot = 0.0
    for kk, rr, n in per.group_by("k", "r").len().iter_rows():
        e = 0.0
        for j in range(rr + 1):  # j decoys among r removed
            pj = comb(rr, j) * pi ** j * (1 - pi) ** (rr - j)
            tp_before, fp_before = kk - j, j
            before = f05(tp_before, fp_before, 0)
            after = f05(kk - rr, 0, rr - j)
            e += pj * (after - before)
        tot += e * n
    print(f"pi={pi}: expected France S1-units {tot:+.0f} -> France F {tot / 259452:+.4f}, LB {tot / 1732544:+.5f}")
cand.write_parquet("/work/dive/fr_sibling_legal_remove.parquet")
