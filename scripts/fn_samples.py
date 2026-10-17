"""Missed true pairs that were candidates (v4, val half B): probability profile, why missed, examples."""
import json, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "src")
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen
work = Path("/work/run4")
cfg = json.loads((work / "val_report_s2.json").read_text())["best"]
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"]
best = pl.read_parquet(work / "val_best_s2.parquet")          # argmax S1 per query (with prob)
chosen = _chosen(best, cfg).with_columns(pl.lit(True).alias("picked"))
tp = lab.filter(pl.col("parent").is_in(s1b.implode())).select(pl.col("rid").alias("q"), pl.col("parent").alias("s1"))
cand = pl.concat([pl.read_parquet(p, columns=["q", "s1"]) for p in sorted((work / "train" / "pruned").glob("*.parquet"))]).join(tp, on=["q", "s1"])
miss = cand.join(chosen, on=["q", "s1"], how="anti").join(best.rename({"s1": "best_s1"}), on="q", how="left")
miss = miss.with_columns((pl.col("best_s1") == pl.col("s1")).alias("argmax_is_true"))
print("missed candidate true pairs:", miss.height)
print(miss.group_by("argmax_is_true").agg(pl.len(), pl.col("prob").median().alias("median_prob_of_argmax")))
m2 = miss.filter(pl.col("argmax_is_true"))
print("when argmax is the true S1, prob histogram:", m2.with_columns(pl.col("prob").cut([0.05, 0.2, 0.4, 0.6, 0.8, 0.95]).alias("b")).group_by("b").len().sort("b").to_dicts())
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["business_name", "business_address", "country"]).with_row_index("rid")
s = miss.sample(20, seed=5).join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")), on="q").join(
    rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("t_name"), pl.col("business_address").alias("t_addr")), on="s1")
with pl.Config(tbl_rows=20, fmt_str_lengths=48, tbl_width_chars=240):
    print(s.select("argmax_is_true", "prob", "q_name", "t_name", "q_addr", "t_addr"))
