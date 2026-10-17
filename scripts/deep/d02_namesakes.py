"""Deep dive 2: empty-address true copies in train and their namesake S1s (same country, same normalized name key).
Prints group-size stats and raw examples: copy name, true S1, other namesakes, and each namesake's copy counts.

uv run --no-project --with polars==1.44.2 python scripts/deep/d02_namesakes.py
"""
import random
import re
import unicodedata

import polars as pl

def key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", s))

S = {s: pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)}
tp = pl.read_parquet("local_data/train_pairs.parquet")
s1 = S[1].with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
rec = pl.concat([S[2], S[3]]).filter(pl.col("business_address").fill_null("").str.strip_chars() == "")
rec = rec.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k")).join(tp, left_on="entity_id", right_on="m", how="left")
print("empty-address records", rec.height, "copies", rec["s1"].is_not_null().sum())
grp = s1.group_by("country", "k").agg(pl.len().alias("n_ns"), pl.col("entity_id").alias("ids"))
r = rec.join(grp, on=["country", "k"], how="left").with_columns(pl.col("n_ns").fill_null(0))
r = r.with_columns(pl.col("ids").list.contains(pl.col("s1")).fill_null(False).alias("true_in_group"))
print(r.group_by(pl.col("n_ns").clip(0, 6)).agg(pl.len(), pl.col("s1").is_not_null().mean().round(4).alias("copy"),
      pl.col("true_in_group").mean().round(4).alias("true_in_exact_key_group")).sort("n_ns"))
cnt = tp.group_by("s1").len().rename({"len": "ncop"})
info = s1.join(cnt, left_on="entity_id", right_on="s1", how="left").fill_null(0)
nm = dict(zip(info["entity_id"], info["business_name"])); ad = dict(zip(info["entity_id"], info["business_address"])); nc = dict(zip(info["entity_id"], info["ncop"]))
ex = r.filter((pl.col("n_ns") >= 2) & pl.col("true_in_group")).sample(25, seed=3)
for row in ex.iter_rows(named=True):
    print("\nCOPY", row["entity_id"], repr(row["business_name"]), row["country"])
    for e in row["ids"]:
        print("   ", "TRUE" if e == row["s1"] else "    ", e, repr(nm[e]), "|", ad[e], "| copies", nc[e])
