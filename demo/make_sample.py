"""Cut a small, realistic slice of the training data for the classroom demo.

A geographic slice keeps the hard cases intact: every Source 1 business in a few cities, all of their true copies
(even ones whose address is empty or garbled), and every other Source 2/3 record in those cities (decoys, plus copies
of businesses outside the slice, which become unmatched records just like in the test set).

python demo/make_sample.py --src local_data --out local_data/demo/data
"""
import argparse
from pathlib import Path

import polars as pl

CITIES = {"US": ["austin", "tucson"], "India": ["rajkot", "vadodara"]}

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=Path("local_data"))
ap.add_argument("--out", type=Path, default=Path("local_data/demo/data"))
a = ap.parse_args()


def in_cities(lf: pl.LazyFrame) -> pl.LazyFrame:
    addr = pl.col("business_address").str.to_lowercase()
    cond = pl.lit(False)
    for country, cities in CITIES.items():
        cond = cond | ((pl.col("country") == country) & addr.str.contains(r"\b(" + "|".join(cities) + r")\b"))
    return lf.filter(cond)


s1 = in_cities(pl.scan_parquet(a.src / "train_source1.parquet")).collect()
gt = (pl.scan_parquet(a.src / "train_pairs.parquet")
      .filter(pl.col("s1").is_in(s1["entity_id"].implode())).collect())
a.out.mkdir(parents=True, exist_ok=True)
s1.write_parquet(a.out / "train_source1.parquet")
keep = set(gt["m"].to_list())
for s in (2, 3):
    lf = pl.scan_parquet(a.src / f"train_source{s}.parquet")
    df = pl.concat([in_cities(lf), lf.filter(pl.col("entity_id").is_in(list(keep)))]).unique("entity_id").collect()
    df.sort("entity_id").write_parquet(a.out / f"train_source{s}.parquet")
    print(f"source {s}: {df.height} records")
gt.write_parquet(a.out / "train_pairs.parquet")
print(f"source 1: {s1.height} businesses ({s1['country'].value_counts().sort('country').rows()}); "
      f"true pairs {gt.height}")
