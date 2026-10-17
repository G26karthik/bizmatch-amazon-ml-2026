"""Deep dive 39: unassigned empty-address records (fr26) whose nkey equals that of exactly one S1 of the country
(unique exact name), split by n_oth_addr (other same-nkey records with a non-empty address not assigned to that S1).
Train truth, all visible (d09 analogue without hiding): P(right) 0.987 (n_oth 0), 0.94 (n_oth 1). Writes candidates.

uv run --no-project --with polars==1.44.2 python scripts/deep/d39_unique_exact_empty.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    t = re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))
    return " ".join(sorted(set(t) - LEG))


for split in ("train", "test"):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    rec = rec.with_columns(pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"), pl.col("entity_id").str.slice(0, 2).alias("src"),
                           (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
    vis = rec.filter(pl.col("src") == "S1").group_by("country", "k").agg(pl.len().alias("n_vis"), pl.col("entity_id").first().alias("X"))
    if split == "train":
        tp = pl.read_parquet("local_data/train_pairs.parquet").rename({"m": "entity_id", "s1": "to"})
    else:
        g = pl.read_csv("output/v28q_fr26/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
        tp = g.select(pl.col("source1_entity_id").alias("to"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("entity_id")).explode("entity_id").filter(pl.col("entity_id") != "")
    m = rec.filter(pl.col("src") != "S1").join(tp, on="entity_id", how="left").join(vis, on=["country", "k"], how="left").filter(pl.col("n_vis") == 1)
    oth = m.filter(~pl.col("empty") & (pl.col("to").is_null() | (pl.col("to") != pl.col("X")))).group_by("country", "k").agg(pl.len().alias("n_oth"))
    e = m.filter(pl.col("empty")).join(oth, on=["country", "k"], how="left").with_columns(pl.col("n_oth").fill_null(0))
    if split == "train":
        print("TRAIN:", e.group_by("country", pl.col("n_oth").clip(0, 2)).agg(pl.len(), (pl.col("to") == pl.col("X")).fill_null(False).mean().round(4).alias("P_right")).sort("country", "n_oth"))
    else:
        u = e.filter(pl.col("to").is_null())
        print("TEST unassigned:", u.group_by("country", pl.col("n_oth").clip(0, 2)).len().sort("country", "n_oth"))
        u.filter(pl.col("n_oth") <= 1).select(pl.col("entity_id").alias("m"), pl.col("X").alias("s1"), "country", pl.col("n_oth").cast(pl.String).alias("kind")).write_parquet(
            "local_data/work/unique_exact_empty_test.parquet")
