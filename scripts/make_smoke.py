"""Build a small consistent subset of /work/data for pipeline smoke tests."""
import sys
from pathlib import Path

import polars as pl

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
gt = pl.read_parquet(src / "train_ground_truth.parquet").sample(20000, seed=0)
pairs = pl.read_parquet(src / "train_pairs.parquet").filter(pl.col("s1").is_in(gt["source1_entity_id"]))
gt.write_parquet(dst / "train_ground_truth.parquet")
pairs.write_parquet(dst / "train_pairs.parquet")
pl.read_parquet(src / "train_source1.parquet").filter(pl.col("entity_id").is_in(gt["source1_entity_id"])).write_parquet(dst / "train_source1.parquet")
for s in (2, 3):
    d = pl.read_parquet(src / f"train_source{s}.parquet")
    keep = d.filter(pl.col("entity_id").is_in(pairs["m"]))
    pl.concat([keep, d.sample(5000, seed=s)]).unique("entity_id").write_parquet(dst / f"train_source{s}.parquet")
for s, n in ((1, 5000), (2, 20000), (3, 20000)):
    pl.read_parquet(src / f"test_source{s}.parquet").sample(n, seed=s).write_parquet(dst / f"test_source{s}.parquet")
print("smoke data ok")
