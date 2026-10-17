"""Where are the remaining missed matches (v15 blend on run11): inside/outside the cross-encoder zone, ranks, probs."""
import json, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "src")
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen, _load
work = Path("/work/run11")
cfg = json.loads((work / "val_report_blend.json").read_text())["best"]
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].cast(pl.UInt32)
best = pl.read_parquet(work / "val_best_blend.parquet")
ch = _chosen(best, cfg).with_columns(pl.lit(True).alias("picked"))
tp = lab.filter(pl.col("parent").is_not_null()).select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32).alias("s1")).filter(pl.col("s1").is_in(s1b.implode()))
d = _load(work, "train").select("q", "s1", "pA").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
d = d.with_columns(pl.col("pA").rank("ordinal", descending=True).over("q").alias("rankA"))
ce = pl.read_parquet("/work/ce_out3/ce_scores_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
sv = pl.read_parquet(work / "scores_val.parquet").select("q", "s1", "p3")
miss = tp.join(d, on=["q", "s1"], how="inner").join(ch, on=["q", "s1"], how="anti")
miss = miss.join(ce, on=["q", "s1"], how="left").join(sv, on=["q", "s1"], how="left").join(best.rename({"s1": "best_s1", "prob": "best_prob"}), on="q", how="left")
miss = miss.with_columns(pl.col("pce").is_not_null().alias("in_ce_zone"), (pl.col("best_s1") == pl.col("s1")).alias("argmax_true"))
print("missed candidate true pairs:", miss.height)
print(miss.group_by("in_ce_zone", "argmax_true").agg(pl.len(), pl.col("pA").median().alias("med_pA"), pl.col("rankA").median().alias("med_rankA"),
      pl.col("pce").median().alias("med_pce"), pl.col("best_prob").median().alias("med_best_prob")).sort("len", descending=True))
print("out-of-zone reasons:", miss.filter(~pl.col("in_ce_zone")).select((pl.col("rankA") > 3).sum().alias("rank>3"), (pl.col("pA") >= 0.98).sum().alias("pA>=0.98")).to_dicts())
