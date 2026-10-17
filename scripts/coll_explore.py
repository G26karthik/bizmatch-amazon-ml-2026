"""Empty-address name collisions on train truth: S2/S3 records with an empty address whose normalised name key
(lower-case alnum tokens, legal/stop words removed) matches 2+ S1 names in the same country. Prints raw examples
(record + every candidate S1, true one marked) and basic counts, to look for raw-level cues (case, accents,
punctuation, legal form) that identify the true S1.

uv run --no-project --with polars python scripts/coll_explore.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def key(s):
    t = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", t)) - LEG))


D = "local_data"
s1 = pl.read_parquet(f"{D}/train_source1.parquet")
oth = pl.concat([pl.read_parquet(f"{D}/train_source2.parquet"), pl.read_parquet(f"{D}/train_source3.parquet")], how="vertical_relaxed")
print(s1.columns, s1.height, oth.height)
gt = pl.read_parquet(f"{D}/train_pairs.parquet")
emp = oth.filter(pl.col("business_address").is_null() | (pl.col("business_address").str.strip_chars() == ""))
print("empty-address S2/S3 records:", emp.height, "of", oth.height)
emp = emp.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
s1k = s1.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
cnt = s1k.group_by("country", "k").agg(pl.len().alias("n_s1"))
emp = emp.join(cnt, on=["country", "k"], how="left").with_columns(pl.col("n_s1").fill_null(0))
emp = emp.join(gt.rename({"m": "entity_id", "s1": "true_s1"}), on="entity_id", how="left")
tk = s1k.select(pl.col("entity_id").alias("true_s1"), pl.col("k").alias("true_k"))
emp = emp.join(tk, on="true_s1", how="left")
print(emp.group_by(pl.col("n_s1").clip(0, 5)).agg(pl.len(), pl.col("true_s1").is_not_null().mean().round(3).alias("has_parent"),
      (pl.col("true_k") == pl.col("k")).mean().round(3).alias("parent_same_key")).sort("n_s1"))
ex = emp.filter((pl.col("n_s1") >= 2) & (pl.col("true_k") == pl.col("k"))).sample(n=25, seed=1)
s1map = s1k.select("entity_id", "business_name", "business_address", "country", "k")
for r in ex.iter_rows(named=True):
    print(f"\nREC {r['entity_id']} [{r['country']}] {r['business_name']!r}  (true {r['true_s1']})")
    for c in s1map.filter((pl.col("k") == r["k"]) & (pl.col("country") == r["country"])).head(8).iter_rows(named=True):
        mark = "  <== TRUE" if c["entity_id"] == r["true_s1"] else ""
        print(f"    {c['entity_id']:>14} {c['business_name']!r:45} | {c['business_address']!r}{mark}")
emp.write_parquet("local_data/work/coll_emp.parquet")
