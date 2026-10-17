"""Deep dive 11: recall budget for ALL S2/S3 records by class (empty address x n_vis bucket), US/India.
Train (19% hiding emulated): P(copy of a visible S1 | class). Test: records per class and share assigned in fr16.
Missing = N_test * P_train - assigned (upper bound of recoverable true copies if assigned ones were all right).

uv run --no-project --with polars==1.44.2 python scripts/deep/d11_recall_budget.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
SEED, HIDE = 42, 19
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    t = re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))
    return " ".join(sorted(set(t) - LEG))


def load(split):
    r = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)])
    return r.with_columns(pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"),
                          (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))


def classes(r, vis):
    nv = vis.group_by("country", "k").len().rename({"len": "n_vis"})
    return r.filter(pl.col("src") != 1).join(nv, on=["country", "k"], how="left").with_columns(pl.col("n_vis").fill_null(0).clip(0, 3))


tr = load("train")
tr = tr.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
vis = tr.filter((pl.col("src") == 1) & ~pl.col("hidden"))
tp = pl.read_parquet("local_data/train_pairs.parquet").filter(pl.col("s1").is_in(vis["entity_id"].implode()))
c = classes(tr, vis).with_columns(pl.col("entity_id").is_in(tp["m"].implode()).alias("copy_vis"))
rate = c.group_by("country", "empty", "n_vis").agg(pl.len().alias("n_train"), pl.col("copy_vis").mean().alias("p_copy"))
te = load("test")
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
ct = classes(te, te.filter(pl.col("src") == 1)).with_columns(pl.col("entity_id").is_in(assigned.implode()).alias("asg"))
t = ct.group_by("country", "empty", "n_vis").agg(pl.len().alias("n_test"), pl.col("asg").sum().alias("assigned"))
t = t.join(rate, on=["country", "empty", "n_vis"], how="left").with_columns((pl.col("n_test") * pl.col("p_copy")).round(0).alias("exp_copies"))
t = t.with_columns((pl.col("exp_copies") - pl.col("assigned")).alias("missing"), pl.col("p_copy").round(4)).sort("country", "empty", "n_vis")
print(t)
print(t.group_by("country").agg(pl.col("n_test").sum(), pl.col("exp_copies").sum(), pl.col("assigned").sum(), pl.col("missing").sum()))
