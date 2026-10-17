"""Record-level true-copy flags (from the generator-rules dive): alias phrase, web-form name, source-inconsistent
address style. (1) Flag rates in train vs test per country/source: decoys never carry them, so the test/train ratio
tells whether test's extra unmatched records are decoys (ratio ~0.60/0.74) or copies of absent S1s (~0.80/0.74).
(2) Precision on raw training labels. (3) How many flagged records v27 leaves unmatched on validation (true rate) and test."""
import json, re, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "src")
from ber.pipeline import labels
from ber.stage2 import _chosen
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
ALIAS = r"(?i)\b(formerly|d/?b/?a|a\.?k\.?a\.?|f/?k/?a|n[ée]e)\b"
WEB = r"(?i)(\.com\b|\bwww\b|@|\|)"
STATE2 = r",\s*[A-Z]{2}\s*$"
def flags(df):
    addr = pl.col("business_address").fill_null("")
    letters = addr.str.replace_all(r"[^A-Za-z]", "")
    upper = (letters.str.len_chars() > 0) & (letters == letters.str.to_uppercase())
    return df.with_columns(pl.col("business_name").str.contains(ALIAS).alias("f_alias"),
                           pl.col("business_name").str.contains(WEB).alias("f_web"),
                           (~pl.col("addr_empty") & (((pl.col("src") == 2) & ~upper) | ((pl.col("src") == 3) & (pl.col("country") == "US") & addr.str.contains(STATE2)))).alias("f_style"))
cols = ["entity_id", "business_name", "business_address", "country", "src", "addr_empty"]
tr = flags(pl.read_parquet("/work/base/train/records.parquet", columns=cols).with_row_index("rid").filter(pl.col("src") != 1))
te = flags(pl.read_parquet("/work/base/test/records.parquet", columns=cols).with_row_index("rid").filter(pl.col("src") != 1))
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("m").alias("entity_id")).with_columns(pl.lit(True).alias("raw_matched"))
tr = tr.join(gt, on="entity_id", how="left").with_columns(pl.col("raw_matched").fill_null(False))
F = ["f_alias", "f_web", "f_style"]
anyf = pl.any_horizontal(F).alias("f_any")
tr = tr.with_columns(anyf); te = te.with_columns(anyf)
print("flag rates (share of S2/S3 records) and train precision (raw match rate):")
a = tr.group_by("country", "src").agg(pl.len().alias("n_train"), *[pl.col(f).mean().alias(f"{f}_train") for f in F + ["f_any"]],
                                      pl.col("raw_matched").filter(pl.col("f_any")).mean().alias("prec_any_train"), pl.col("raw_matched").mean().alias("match_rate_train"))
b = te.group_by("country", "src").agg(pl.len().alias("n_test"), *[pl.col(f).mean().alias(f"{f}_test") for f in F + ["f_any"]])
t = a.join(b, on=["country", "src"], how="full", coalesce=True).with_columns((pl.col("f_any_test") / pl.col("f_any_train")).alias("ratio_any_test_train")).sort("country", "src")
print(t.select("country", "src", "n_train", "n_test", "match_rate_train", "f_alias_train", "f_alias_test", "f_web_train", "f_web_test", "f_style_train", "f_style_test", "prec_any_train", "ratio_any_test_train"))
# v27 decisions for flagged records
cfg = json.load(open("/work/run24/val_report_run24_both.json"))["best"]
bt = pl.read_parquet("/work/run24/test_best_run24_both.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
selt = _chosen(bt, cfg).select("q").with_columns(pl.lit(True).alias("sel"))
x = te.select(pl.col("rid").cast(pl.UInt32).alias("q"), "country", "f_any", "f_alias", "f_web", "addr_empty").join(bt.select("q", "prob"), on="q", how="left").join(selt, on="q", how="left").with_columns(pl.col("sel").fill_null(False))
print("TEST (v27): flagged records, share selected, and not-selected with a candidate by best prob band")
print(x.filter(pl.col("f_any")).group_by("country").agg(pl.len(), pl.col("sel").mean().alias("selected"), (~pl.col("sel") & pl.col("prob").is_not_null()).sum().alias("unselected_with_cand"),
      (~pl.col("sel") & (pl.col("prob") >= 0.3)).sum().alias("unsel_p>=0.3"), (~pl.col("sel") & (pl.col("prob") >= 0.5)).sum().alias("unsel_p>=0.5"), pl.col("prob").is_null().sum().alias("no_candidate")).sort("country"))
print("TEST (v27): unflagged records for comparison"); print(x.filter(~pl.col("f_any")).group_by("country").agg(pl.len(), pl.col("sel").mean().alias("selected")).sort("country"))
# validation: flagged records of validation groups, not selected -> true rate (after hiding) and raw
lab = labels(Path("/work/data"), Path("/work/base"))
bv = pl.read_parquet("/work/run24/val_best_run24_both.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
selv = _chosen(bv, cfg).select("q").with_columns(pl.lit(True).alias("sel"))
v = tr.select(pl.col("rid").cast(pl.UInt32).alias("q"), "country", "f_any", "raw_matched", "addr_empty").join(bv, on="q", how="inner").join(selv, on="q", how="left").with_columns(pl.col("sel").fill_null(False)) \
      .join(lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32), "is_val"), on="q").filter(pl.col("is_val")) \
      .with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
print("VALIDATION (v27): flagged vs unflagged, not-selected records with p>=0.3: true rate (best pair correct) and raw match rate")
print(v.filter(~pl.col("sel") & (pl.col("prob") >= 0.3)).group_by("f_any", "addr_empty").agg(pl.len(), pl.col("y").mean().alias("best_pair_true"), pl.col("raw_matched").mean().alias("raw_matched_any_S1")).sort("f_any", "addr_empty"))
