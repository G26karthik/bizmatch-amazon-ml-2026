"""Generator dive prep (worker): one pass over records to build
  tp_s.parquet   300k sampled true pairs with raw + normalised text of both sides
  best2.parquet  every S2/S3 record's best S1 candidate over the FULL S1 pool (run2 stage-A, no hidden S1s) + label
  allq_{C}.parquet  best2 joined with text of both sides, per country (for vectorised rule mining)
"""
import sys

import polars as pl

R = "/work/run4/train/records.parquet"
COLS = ["business_name", "business_address", "src", "country", "name_n", "addr_w", "house", "postal", "addr_empty", "name_nonascii"]
tp = pl.read_parquet("/work/dive/tp.parquet").with_columns(pl.col("s1r").cast(pl.UInt32), pl.col("mr").cast(pl.UInt32))
par = tp.select(pl.col("mr").alias("q"), pl.col("s1r").alias("parent"))

# ---- best candidate over the full pool (run2 has no hidden S1s) -- same rid order? check entity ids on a stride
# (checked 2026-09-26: entity_id at every 997th rid identical in run2 and run4 records)
F = ["q", "s1", "pA", "name_tsr", "name_ratio", "house_eq", "house_miss", "house_absdiff", "street_tsr", "addr_tsr",
     "postal_eq", "first_tok_eq", "q_addr_empty", "country"]
pr = pl.scan_parquet("/work/run2/train/pruned/*.parquet").select(F)
mx = pr.group_by("q").agg(pl.col("pA").max().alias("_mx"), pl.len().alias("ncand"))
best = pr.join(mx, on="q").filter(pl.col("pA") == pl.col("_mx")).unique("q", keep="first").drop("_mx").collect()
best = best.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ids = pl.scan_parquet(R).select("src", "country").with_row_index("rid").filter(pl.col("src") > 1).collect()
best = ids.rename({"rid": "q", "country": "qc"}).join(best.drop("country"), on="q", how="left") \
          .join(par, on="q", how="left").with_columns(
    pl.col("parent").is_not_null().alias("matched"),
    (pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
best.write_parquet("/work/dive/best2.parquet")
print(best.group_by("qc", "src").agg(pl.len(), pl.col("matched").mean(), pl.col("s1").is_null().mean().alias("no_cand"),
                                      pl.col("y").mean(), (pl.col("y") | ~pl.col("matched")).mean().alias("best_is_parent_or_decoy")).sort("qc", "src"))
sys.stdout.flush()

# ---- text (30% sample of S2/S3 records with a candidate; tp_s2 is built by g1_fetch.py)
samp = best.filter(pl.col("s1").is_not_null() & (pl.col("q").hash(3) % 10 < 3))
need = pl.concat([samp["q"], samp["s1"]]).unique().to_frame("rid")
rec = pl.scan_parquet(R).select(COLS).with_row_index("rid").join(need.lazy(), on="rid", how="semi").collect()
t1 = rec.select(pl.col("rid").alias("s1r"), *[pl.col(c).alias(c + "_1") for c in COLS if c not in ("src", "country", "business_address")])
t2 = rec.select(pl.col("rid").alias("mr"), *[pl.col(c).alias(c + "_2") for c in COLS if c not in ("src", "country", "business_address")])
for c in ("US", "India"):
    b = samp.filter(pl.col("qc") == c).rename({"q": "mr", "s1": "s1r"})
    b = b.join(t1, on="s1r").join(t2, on="mr")
    b.write_parquet(f"/work/dive/allq_{c}.parquet")
    print("allq", c, b.height, flush=True)
    del b
