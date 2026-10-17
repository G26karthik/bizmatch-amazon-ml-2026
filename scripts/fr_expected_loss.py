"""Where does France's remaining model-implied loss sit? Accepted pairs (fr_name_ops_audit.parquet, fr13) contribute
expected false positives sum(1 - p); unaccepted best-S1 pairs with p > 0.1 (fr_name_ops_audit_unacc.parquet) contribute
expected misses sum(p). p = blended prob (test_best run28_7q); contested swaps use the rule (vocab word 0.02, non-word 0.95).
Per pattern x address class, with bracket share as a model-free copy marker (copies ~4.2%, decoys ~0.9%).

python scripts/fr_expected_loss.py
"""
import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
cv = pl.read_parquet("/work/lead/fr_contested_vocab.parquet").select("q", "s1", (pl.col("voc1") > 30).alias("vocab"))
res = []
for f, acc in (("/work/lead/fr_name_ops_audit.parquet", True), ("/work/lead/fr_name_ops_audit_unacc.parquet", False)):
    x = pl.read_parquet(f).join(tb, on=["q", "s1"], how="left").join(cv, on=["q", "s1"], how="left")
    x = x.with_columns(pl.when(pl.col("vocab") == True).then(0.02).when(pl.col("vocab") == False).then(0.95)  # noqa: E712
                       .otherwise(pl.col("prob").fill_null(0.9)).alias("p"))
    t = x.group_by("pat", "arel").agg(pl.len().alias("n"), ((1 - pl.col("p")) if acc else pl.col("p")).sum().round(0).alias("exp_err"),
                                      pl.col("p").mean().round(3).alias("mean_p"), pl.col("bracket").mean().round(4).alias("brk"))
    res.append(t.with_columns(pl.lit("accepted_FP" if acc else "unaccepted_FN").alias("kind")))
r = pl.concat(res).sort("exp_err", descending=True)
print("total expected FP (accepted):", r.filter(pl.col("kind") == "accepted_FP")["exp_err"].sum(), " expected FN (unaccepted p>0.1):",
      r.filter(pl.col("kind") == "unaccepted_FN")["exp_err"].sum())
print(r.head(30))
