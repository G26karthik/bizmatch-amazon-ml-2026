"""Deep dive 9: orphan evidence for empty-address records (train truth, 19% S1 hiding emulated as in validation).
Record r (S2/S3, empty address). nkey = sorted name tokens minus legal forms/stopwords. Visible S1s with r's nkey:
n_vis. For n_vis == 1 (unique visible candidate X): P(r belongs to X) split by
  n_oth_addr = number of OTHER S2/S3 records with the same nkey, a non-empty address, not copies of X (at test time:
               records not assigned to X), and whether any of them has an address unlike X's.
Also for n_vis >= 2 the share where r belongs to one of them.

uv run --no-project --with polars==1.44.2 python scripts/deep/d09_orphan_evidence.py
"""
import re
import unicodedata

import polars as pl

SEED, HIDE = 42, 19
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    t = re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))
    return " ".join(sorted(set(t) - LEG))


rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)])
rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
rec = rec.with_columns(pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"),
                       (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
tp = pl.read_parquet("local_data/train_pairs.parquet")
vis = rec.filter((pl.col("src") == 1) & ~pl.col("hidden")).select("entity_id", "country", "k")
m = rec.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("m"), "country", "k", "empty").join(tp, on="m", how="left")
hid = set(rec.filter(pl.col("hidden"))["entity_id"].to_list())
m = m.with_columns(pl.col("s1").is_in(list(hid)).fill_null(False).alias("to_hidden"))
nv = vis.group_by("country", "k").agg(pl.len().alias("n_vis"), pl.col("entity_id").first().alias("X"))
m = m.join(nv, on=["country", "k"], how="left").with_columns(pl.col("n_vis").fill_null(0))
# per (country,k): addressed records not belonging to the unique visible X
oth = m.filter(~pl.col("empty") & (pl.col("s1").is_null() | (pl.col("s1") != pl.col("X")))).group_by("country", "k").agg(pl.len().alias("n_oth_addr"))
e = m.filter(pl.col("empty")).join(oth, on=["country", "k"], how="left").with_columns(pl.col("n_oth_addr").fill_null(0))
print("empty-address records", e.height)
print(e.group_by(pl.col("n_vis").clip(0, 3)).agg(pl.len(), (pl.col("s1").is_not_null() & ~pl.col("to_hidden")).mean().round(4).alias("copy_of_visible"),
      pl.col("to_hidden").mean().round(4).alias("copy_of_hidden"), pl.col("s1").is_null().mean().round(4).alias("no_s1")).sort("n_vis"))
u = e.filter(pl.col("n_vis") == 1).with_columns((pl.col("s1") == pl.col("X")).fill_null(False).alias("isX"))
print("\nunique visible nkey candidate X: P(r in X) by other addressed same-key records not in X")
print(u.group_by(pl.col("n_oth_addr").clip(0, 4)).agg(pl.len(), pl.col("isX").mean().round(4).alias("P_in_X"),
      pl.col("to_hidden").mean().round(4).alias("P_hidden")).sort("n_oth_addr"))
