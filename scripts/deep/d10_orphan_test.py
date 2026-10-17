"""Deep dive 10: orphan evidence on test vs the fr16 decisions. Empty-address S2/S3 records with exactly one visible S1
X sharing the nkey; n_oth_addr = other same-nkey records with a non-empty address not assigned to X. Share assigned
to X / elsewhere / unassigned, by country and n_oth_addr; plus X's current source count (cap room).

uv run --no-project --with polars==1.44.2 python scripts/deep/d10_orphan_test.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    t = re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))
    return " ".join(sorted(set(t) - LEG))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
rec = rec.with_columns(pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"), pl.col("entity_id").str.slice(0, 2).alias("src"),
                       (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.select(pl.col("source1_entity_id").alias("to"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
cnt = a.with_columns(pl.col("m").str.slice(0, 2).alias("src")).group_by("to", "src").len().rename({"len": "n_src"})
vis = rec.filter(pl.col("src") == "S1").group_by("country", "k").agg(pl.len().alias("n_vis"), pl.col("entity_id").first().alias("X"))
m = rec.filter(pl.col("src") != "S1").select("entity_id", "src", "country", "k", "empty").join(a.rename({"m": "entity_id"}), on="entity_id", how="left")
m = m.join(vis, on=["country", "k"], how="left").with_columns(pl.col("n_vis").fill_null(0))
oth = m.filter(~pl.col("empty") & (pl.col("to").is_null() | (pl.col("to") != pl.col("X")))).group_by("country", "k").agg(pl.len().alias("n_oth_addr"))
e = m.filter(pl.col("empty") & (pl.col("n_vis") == 1)).join(oth, on=["country", "k"], how="left").with_columns(pl.col("n_oth_addr").fill_null(0))
e = e.join(cnt.rename({"to": "X"}), on=["X", "src"], how="left").with_columns(pl.col("n_src").fill_null(0))
e = e.with_columns(pl.when(pl.col("to") == pl.col("X")).then(pl.lit("X")).when(pl.col("to").is_not_null()).then(pl.lit("other")).otherwise(pl.lit("none")).alias("st"))
print(e.group_by("country", pl.col("n_oth_addr").clip(0, 2)).agg(pl.len(), (pl.col("st") == "X").mean().round(4).alias("to_X"),
      (pl.col("st") == "other").mean().round(4).alias("to_other"), (pl.col("st") == "none").sum().alias("n_unassigned"),
      ((pl.col("st") == "none") & (((pl.col("src") == "S2") & (pl.col("n_src") >= 5)) | ((pl.col("src") == "S3") & (pl.col("n_src") >= 6)))).sum().alias("unassigned_X_at_cap"))
      .sort("country", "n_oth_addr"))
e.filter((pl.col("st") == "none") & (pl.col("n_oth_addr") == 0)).select("entity_id", "X", "country", "src", "n_src").write_parquet(
    "local_data/work/orphan0_unassigned.parquet")
