"""Agent FR a8: accepted records per S1 (fr25) by S1 address type (house number present / no house / empty) and
multi-tenant status (other S1s with the same normalized address), per country and source."""
import re, unicodedata
from collections import Counter
import polars as pl
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
def norm(s): return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
s1 = pl.read_parquet("local_data/test_source1.parquet", columns=["entity_id","business_address","country"])
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
g = g.with_columns(pl.col("matched_entity_ids").fill_null("")).with_columns(
    pl.col("matched_entity_ids").str.count_matches("S2").alias("n2"), pl.col("matched_entity_ids").str.count_matches("S3").alias("n3"))
at, key = [], []
for a in s1["business_address"].to_list():
    t = re.findall(r"[a-z0-9]+", norm(a))
    key.append(" ".join(sorted(t)))
    at.append("empty" if not t else ("house" if any(x.isdigit() and len(x) < 6 for x in t) else "nohouse"))
s1 = s1.with_columns(pl.Series("atype", at), pl.Series("akey", key))
kc = s1.group_by("country", "akey").len().rename({"len": "tenants"})
s1 = s1.join(kc, on=["country", "akey"]).with_columns(pl.when(pl.col("tenants") > 1).then(pl.lit("multi")).otherwise(pl.lit("single")).alias("mt"))
x = s1.join(g.rename({"source1_entity_id": "entity_id"}), on="entity_id", how="left").fill_null(0)
print(x.group_by("country", "atype", "mt").agg(pl.len().alias("S1s"), pl.col("n2").mean().round(3), pl.col("n3").mean().round(3),
      (pl.col("n2") + pl.col("n3")).mean().round(3).alias("n"), ((pl.col("n2") + pl.col("n3")) == 0).mean().round(4).alias("p0")).sort("country", "atype", "mt"))
