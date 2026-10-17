"""Candidate-set size vs recall trade-off on the pruned training set (run4). Run on worker 1.

For stage-A probability thresholds and per-record top-k caps, report candidates per S1 entity and the share of true
matches kept, so the candidate file can be made as small as possible without hurting matching quality.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import labels  # noqa: E402

work = Path("/work/run4")
lab = labels(Path("/work/data"), work)
n_s1 = lab.filter(pl.col("src") == 1).height
qlab = lab.select(pl.col("rid").alias("q"), "parent")
d = pl.concat([pl.read_parquet(p, columns=["q", "s1", "pA"]) for p in sorted((work / "train" / "pruned").glob("*.parquet"))])
d = d.join(qlab, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
d = d.with_columns(pl.col("pA").rank("ordinal", descending=True).over("q").alias("rk"))
n_true = int(lab["parent"].is_not_null().sum())
print(f"S1 entities {n_s1}, true pairs {n_true}, current pruned pairs {d.height} "
      f"({d.height / n_s1:.2f}/S1), recall {d['y'].sum() / n_true:.4f}")
rows = []
for thr in (0.001, 0.003, 0.01, 0.02, 0.05, 0.1):
    for k in (99, 5, 3, 2):
        s = d.filter((pl.col("pA") >= thr) & (pl.col("rk") <= k))
        rows.append((thr, k, round(s.height / n_s1, 2), round(s["y"].sum() / n_true, 5)))
print(pl.DataFrame(rows, schema=["pA_thr", "topk_per_record", "cand_per_S1", "recall"], orient="row"))
