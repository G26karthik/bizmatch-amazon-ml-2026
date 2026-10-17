"""E00 part 10: the Stage-A band 0.001 <= pA < 0.01 on half-B S1s: how many pairs, how many true, how concentrated."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import SEED, labels  # noqa: E402

work = Path("/work/run28")
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
SB = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1"))
par = lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32))
p = pl.concat([pl.read_parquet(f, columns=["q", "s1", "pA", "q_addr_empty", "house_eq", "name_tsr"]) for f in sorted((work / "train" / "pruned").glob("*.parquet"))])
p = p.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
qmax = p.group_by("q").agg(pl.col("pA").max().alias("q_max_pA"))
b = p.filter((pl.col("pA") < 0.01)).join(SB, on="s1", how="semi").join(par, on="q", how="left").join(qmax, on="q")
b = b.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"), (pl.col("q_max_pA") < 0.01).alias("record_unscored"))
for lo, hi in ((0.001, 0.01), (0.005, 0.01), (0.001, 0.005)):
    x = b.filter((pl.col("pA") >= lo) & (pl.col("pA") < hi))
    print(f"band {lo}-{hi}: pairs {x.height}, true {x['y'].sum()} ({x['y'].mean():.4f}); records with no pair >= 0.01: "
          f"{x.filter(pl.col('record_unscored')).height} pairs, true {x.filter(pl.col('record_unscored'))['y'].sum()}")
x = b.filter(pl.col("pA") >= 0.001)
with pl.Config(tbl_rows=20):
    print(x.group_by("q_addr_empty", "house_eq").agg(pl.len(), pl.col("y").sum().alias("true"), pl.col("y").mean().alias("prec")).sort("q_addr_empty", "house_eq"))
