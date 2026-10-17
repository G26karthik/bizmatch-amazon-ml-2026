"""Tie-breaker: source mix (share of S2 vs S3) and formatting of swap records by category. If copies and decoys come
from different generator paths their source shares may differ; US train truth shows whether they do. France: compares
same-address type-word swaps with same-address generic swaps (copies) and offset type-word swaps (decoys).

python scripts/swap_source_mix.py
"""
import sys

import polars as pl

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
GEN_FR = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
GEN_US = ["center", "services", "service", "partners", "centre"]


def summarize(d, path, label):
    rec = pl.read_parquet(path, columns=["src", "business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    x = d.join(rec.select(pl.col("rid").alias("q"), "src", "business_name", "business_address"), on="q")
    x = x.with_columns((pl.col("business_name") == pl.col("business_name").str.to_uppercase()).alias("upper"),
                       pl.col("business_address").str.contains(r"(?i)hauts-de-france|nouvelle-aquitaine|pays de la loire").alias("region"))
    print(f"\n=== {label}")
    print(x.group_by("cat").agg(pl.len().alias("n"), (pl.col("src") == 2).mean().round(4).alias("share_S2"),
                                pl.col("upper").mean().round(4).alias("upper_name"), pl.col("region").mean().round(4).alias("region_in_addr")).sort("cat"))


tr = ns["prep"]("/work/base/train/records.parquet")
d = ns["swaps"](tr.filter(pl.col("country") == "US"))
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
d = d.with_columns(pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_US)).then(pl.lit("GEN")).otherwise(pl.lit("OTHER")),
                                  pl.when(pl.col("true")).then(pl.lit("TRUE")).otherwise(pl.lit("FALSE"))], separator="|").alias("cat"))
summarize(d, "/work/base/train/records.parquet", "US train truth")
f = pl.read_parquet("/work/dive/fr_swaps.parquet").with_columns(
    pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_FR)).then(pl.lit("GEN")).otherwise(pl.lit("TYPE"))], separator="|").alias("cat"))
summarize(f, "/work/runfr/test/records.parquet", "France test")
# reference: identical name + same address records (certain copies) in France
