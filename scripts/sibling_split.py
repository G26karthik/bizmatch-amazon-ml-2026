"""US train truth: false one-word swaps (sibling decoys) by address relation (same vs house + decoy offset) and swapped-in
word class; tests whether the generator splits sibling decoys between the S1's own house and an offset house at a fixed
ratio (France type-word swaps: same/offset = 0.36 for every type word).

python scripts/sibling_split.py
"""
import sys

import polars as pl

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
tr = ns["prep"]("/work/base/train/records.parquet")
d = ns["swaps"](tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
GEN = ["center", "services", "service", "partners", "centre", "holdings", "group"]
for c in ("US", "India"):
    x = d.filter((pl.col("country") == c) & ~pl.col("true"))
    x = x.with_columns(pl.col("w_in").is_in(GEN).alias("generic"))
    t = x.group_by("generic", "arel").len().pivot(on="arel", index="generic", values="len").with_columns((pl.col("same") / pl.col("offset")).round(3).alias("same_over_offset"))
    print(c, "FALSE swaps:\n", t)
    w = x.filter(~pl.col("generic")).group_by("w_in", "arel").len().pivot(on="arel", index="w_in", values="len").fill_null(0) \
         .with_columns((pl.col("same") + pl.col("offset")).alias("tot")).sort("tot", descending=True).head(12) \
         .with_columns((pl.col("same") / pl.col("offset")).round(3).alias("ratio"))
    print(w)
    y = d.filter((pl.col("country") == c) & pl.col("true"))
    print(c, "TRUE swaps by arel:", y.group_by("arel").len().rows())
