"""Decoy-offset rule on validation (labelled): accepted pairs whose record house = S1 house + generator decoy offset.
Measures precision of those accepted pairs and the half A / half B effect of removing them, per offset set.

python scripts/decoy_offset_val.py --work /work/run28 --tag run28_7q
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("/work/run28"))
ap.add_argument("--tag", default="run28_7q")
a = ap.parse_args()
rec = pl.read_parquet(a.work / "train" / "records.parquet", columns=["house", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
lab = labels(Path("/work/data"), a.work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))
S = {h: s1v.filter(pl.col("half_a") == (h == "A"))["rid"].to_list() for h in "AB"}
parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in parent.items():
    truth.setdefault(p, set()).add(q)
cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
ch = _chosen(pl.read_parquet(a.work / f"val_best_{a.tag}.parquet"), cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
d = ch.join(rec.select(pl.col("rid").alias("q"), pl.col("house").alias("hq")), on="q").join(rec.select(pl.col("rid").alias("s1"), pl.col("house").alias("hs"), "country"), on="s1")
d = d.filter(pl.col("hq").str.contains(r"^\d+$") & pl.col("hs").str.contains(r"^\d+$")) \
     .with_columns((pl.col("hq").cast(pl.Int64) - pl.col("hs").cast(pl.Int64)).alias("off"))
d = d.with_columns(pl.Series("true", [parent.get(q) == s for q, s in zip(d["q"].to_list(), d["s1"].to_list())]))
print("accepted validation pairs by house offset (record - S1):")
print(d.filter(pl.col("off").is_between(-3, 25) & (pl.col("off") != 0)).group_by("off").agg(pl.len().alias("n"), pl.col("true").mean().round(3).alias("true_rate")).sort("off"))


def macro(chs, half):
    pm = {}
    for q, s in chs.iter_rows():
        pm.setdefault(s, set()).add(q)
    return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S[half]]))


base = {h: macro(ch, h) for h in "AB"}
for name, offs in (("3-21 decoy offsets", {3, 4, 5, 7, 9, 11, 13, 21}), ("all decoy offsets", {1, 2, 3, 4, 5, 7, 9, 11, 13, 21})):
    rm = d.filter(pl.col("off").is_in(list(offs)))
    new = ch.join(rm.select("q", "s1"), on=["q", "s1"], how="anti")
    g = {h: macro(new, h) - base[h] for h in "AB"}
    print(f"REMOVE {name}: {rm.height} pairs (true {rm['true'].sum()}, precision of removed-as-decoy {1 - rm['true'].mean():.3f}) gain A {g['A']:+.6f} B {g['B']:+.6f}")
