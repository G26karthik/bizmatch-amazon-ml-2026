"""Are the French copy/decoy markers confounded with the model probability? US/India validation truth:
(1) cap: among FALSE best-S1 pairs, P(prob > t) at S1-source slots with 2-3 other chosen records vs at the cap (5 S2 / 6 S3).
    If the model lowers probabilities at full S1s, the cap test split by probability is biased.
(2) bracket: P(prob > t | record adds a bracket the S1 lacks) vs without, separately for true and false pairs.
    If brackets raise probabilities of false pairs, the bracket estimate split by probability is biased.

python scripts/val_confound.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
work = Path("/work/run28")
lab = labels(Path("/work/data"), work)
parent = lab.select(pl.col("rid").alias("q"), "parent").with_columns(pl.col("q").cast(pl.UInt32), pl.col("parent").cast(pl.UInt32))
vb = pl.read_parquet(work / "val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1", "prob")
vb = vb.join(parent, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
ch = _chosen(pl.read_parquet(work / "val_best_run28_7q.parquet"), {"mode": "efopt", "miss": 0.0, "gamma": 1.5}) \
       .select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["business_name", "src"]).with_row_index("q").with_columns(pl.col("q").cast(pl.UInt32))
ch = ch.join(rec.select("q", "src"), on="q")
vb = vb.join(rec.select("q", "src"), on="q")
cnt = ch.group_by("s1", "src").agg(pl.len().alias("n_ch"))
vb = vb.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_ch").fill_null(0)) \
       .join(ch.select("q", "s1").with_columns(pl.lit(1).alias("self")), on=["q", "s1"], how="left") \
       .with_columns((pl.col("n_ch") - pl.col("self").fill_null(0)).alias("n_other"))
capc = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
vb = vb.with_columns(pl.when(capc).then(pl.lit("cap")).when(pl.col("n_other").is_between(2, 3)).then(pl.lit("n23"))
                     .when(pl.col("n_other") == 4).then(pl.lit("n4")).otherwise(pl.lit("other")).alias("slot"))
print("(1) cap: prob distribution by slot, false and true best-S1 pairs")
print(vb.filter(pl.col("slot") != "other").group_by("y", "slot").agg(pl.len().alias("n"), (pl.col("prob") > 0.5).mean().round(4).alias("p>0.5"),
      (pl.col("prob") > 0.99).mean().round(4).alias("p>0.99"), (pl.col("prob") > 0.999).mean().round(4).alias("p>0.999"),
      pl.col("prob").mean().round(4).alias("mean_p")).sort("y", "slot"))
nm = rec["business_name"].to_list()
br = [(nm[q].count("[") + nm[q].count("(")) > (nm[s].count("[") + nm[s].count("(")) for q, s in zip(vb["q"].to_list(), vb["s1"].to_list())]
vb = vb.with_columns(pl.Series("bracket", br))
print("\n(2) bracket: prob distribution by bracket, true and false pairs")
print(vb.group_by("y", "bracket").agg(pl.len().alias("n"), (pl.col("prob") > 0.5).mean().round(4).alias("p>0.5"),
      (pl.col("prob") > 0.99).mean().round(4).alias("p>0.99"), (pl.col("prob") > 0.999).mean().round(4).alias("p>0.999"),
      pl.col("prob").mean().round(4).alias("mean_p")).sort("y", "bracket"))
print("\nbracket rate by truth among pairs with prob > 0.5:",
      vb.filter(pl.col("prob") > 0.5).group_by("y").agg(pl.col("bracket").mean().round(4)).sort("y").rows())
