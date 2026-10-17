"""Deep dive 3: France vs US/India structure of the current best predictions (v27) on test, next to the same statistics
for TRUE pairs on validation (US/India labels). Deviations point to where France errors are.
Per country: selected matches per S1, zero-match S1s, and among selected pairs: house relation, empty address,
exact / brand-like names, legal-suffix change, source mix."""
import json, re, sys
from pathlib import Path
import polars as pl
from unidecode import unidecode
sys.path.insert(0, "src")
from ber.stage2 import _chosen
from ber.pipeline import labels
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
SUF = re.compile(r"(private limited|pvt\.? ?ltd\.?|pvt|limited|ltd\.?|l\.?l\.?c\.?|inc\.?|incorporated|corp\.?|corporation|co\.|company|llp|p\.?c\.?|pllc"
                 r"|s\.?a\.?r\.?l\.?|s\.?a\.?s\.?u?|s\.?a\.?|e\.?u\.?r\.?l\.?|s\.?c\.?i\.?|e\.?i\.?|snc)\s*[\]\)]?\s*$", re.I)
def suf(s):
    m = SUF.search(unidecode(s or "").strip()); return re.sub(r"[^a-z]", "", m.group(1).lower()) if m else ""
cfg = json.load(open("/work/run24/val_report_run24_both.json"))["best"]
cols = ["q", "s1", "house_eq", "house_miss", "q_addr_empty", "name_tsr", "street_tsr"]
def stats(pairs, rec, feats, label):
    p = pairs.join(feats, on=["q", "s1"], how="left").join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("qn"), pl.col("src").alias("qsrc"), "country", "addr_empty"), on="q") \
             .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("sn")), on="s1")
    p = p.with_columns(pl.Series("suf_change", [suf(a) != suf(b) for a, b in zip(p["qn"].to_list(), p["sn"].to_list())]),
                       (pl.col("qn").str.to_lowercase() == pl.col("sn").str.to_lowercase()).alias("name_exact_ci"))
    return p.group_by("country").agg(pl.len().alias(f"{label}_pairs"), pl.col("addr_empty").mean().alias("q_empty_addr"),
            (pl.col("house_eq").fill_null(False)).mean().alias("same_house"), (~pl.col("house_eq").fill_null(True) & ~pl.col("house_miss").fill_null(True)).mean().alias("diff_house"),
            pl.col("name_exact_ci").mean().alias("name_exact"), (pl.col("name_tsr") < 40).mean().alias("brand_like"),
            pl.col("suf_change").mean().alias("legal_suffix_change"), (pl.col("qsrc") == 3).mean().alias("from_S3")).with_columns(pl.lit(label).alias("set")).sort("country")
# test, selected pairs of v27
rte = pl.read_parquet("/work/base/test/records.parquet", columns=["business_name", "src", "country", "addr_empty"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
bt = pl.read_parquet("/work/run24/test_best_run24_both.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
selt = _chosen(bt, cfg)
ft = pl.concat([pl.scan_parquet("/work/run29/test/pruned/*.parquet").select(cols).collect()]).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
st = stats(selt, rte, ft, "test_selected")
s1t = rte.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country")
per = s1t.join(selt.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
recs = rte.filter(pl.col("src") != 1).group_by("country").len().rename({"len": "records"})
print("TEST (v27): per-S1 structure")
print(per.group_by("country").agg(pl.len().alias("S1"), pl.col("len").mean().alias("matches_per_S1"), (pl.col("len") == 0).mean().alias("zero_match_S1"),
      (pl.col("len") >= 6).mean().alias("S1_with_6plus")).join(recs, on="country").with_columns((pl.col("records") / pl.col("S1")).alias("records_per_S1"),
      (pl.col("matches_per_S1") / (pl.col("records") / pl.col("S1"))).alias("share_records_matched")).sort("country"))
# validation truth (US/India): true pairs of validation S1s, no hiding bias in the pair itself
lab = labels(Path("/work/data"), Path("/work/base"))
rtr = pl.read_parquet("/work/base/train/records.parquet", columns=["business_name", "src", "country", "addr_empty"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
truth = lab.filter(pl.col("parent").is_not_null() & pl.col("is_val")).select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32).alias("s1"))
fv = pl.scan_parquet("/work/run29/train/pruned/*.parquet").select(cols).collect().with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
sv_ = stats(truth, rtr, fv, "val_true")
print("SELECTED test pairs (v27) vs TRUE validation pairs:")
print(pl.concat([sv_, st], how="diagonal_relaxed").select("set", "country", pl.col("^.*_pairs$"), "q_empty_addr", "same_house", "diff_house", "name_exact", "brand_like", "legal_suffix_change", "from_S3"))
tv = lab.filter((pl.col("src") == 1) & pl.col("is_val")).select(pl.col("rid").cast(pl.UInt32).alias("s1")).join(rtr.select(pl.col("rid").alias("s1"), "country"), on="s1")
pv = tv.join(truth.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
print("VALIDATION truth per S1:"); print(pv.group_by("country").agg(pl.col("len").mean().alias("true_matches_per_S1"), (pl.col("len") == 0).mean().alias("zero_match_S1"), (pl.col("len") >= 6).mean().alias("S1_with_6plus")))
