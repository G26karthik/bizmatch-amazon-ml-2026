"""Deep dive 7: empty-address records in test by country: count, share assigned in fr16, namesake group size of the
record's exact name key among S1s of its country; and per-S1 namesake multiplicity.

uv run --no-project --with polars==1.44.2 python scripts/deep/d07_empty_by_country.py
"""
import re
import unicodedata

import polars as pl

def key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return " ".join(re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch))))

rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
rec = rec.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"), pl.col("entity_id").str.slice(0, 2).alias("src"))
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(e for m in g["matched_entity_ids"].fill_null("").to_list() for e in m.split(",") if e)
s1 = rec.filter(pl.col("src") == "S1")
ns = s1.group_by("country", "k").len().rename({"len": "n_ns"})
print("S1 namesake multiplicity (share of S1s whose name key is shared):")
print(s1.join(ns, on=["country", "k"]).group_by("country").agg((pl.col("n_ns") >= 2).mean().round(4).alias("shared"), pl.col("n_ns").mean().round(3).alias("mean_group")).sort("country"))
r = rec.filter(pl.col("src") != "S1").with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"),
                                                   pl.col("entity_id").is_in(list(assigned)).alias("assigned"))
r = r.join(ns, on=["country", "k"], how="left").with_columns(pl.col("n_ns").fill_null(0))
print(r.group_by("country").agg(pl.len(), pl.col("empty").mean().round(4).alias("empty_share"), pl.col("assigned").mean().round(4).alias("assigned_all"),
      pl.col("assigned").filter(pl.col("empty")).mean().round(4).alias("assigned_empty"), pl.col("empty").sum().alias("n_empty"),
      (pl.col("empty") & ~pl.col("assigned")).sum().alias("empty_unassigned")).sort("country"))
e = r.filter(pl.col("empty"))
pl.Config.set_tbl_rows(-1)
print(e.group_by("country", pl.col("n_ns").clip(0, 5)).agg(pl.len(), pl.col("assigned").mean().round(4)).sort("country", "n_ns"))
