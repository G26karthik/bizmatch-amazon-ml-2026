"""Decision-strictness variants of a blended submission (probabilities unchanged, gamma of the expected-F0.5 rule
varied), with the half-B validation score of each rule for reference. Used to calibrate the decoy prior on the
leaderboard: validation cannot show how decoy-heavy the test set is.

python scripts/gamma_variants.py --work /work/run20 --tag run20_large --out /work/out_gamma --gammas 1.0 2.0 2.5 3.0
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen, _macro_hard, _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/work/data"))
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--gammas", type=float, nargs="+", required=True)
a = ap.parse_args()
cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
lab = labels(a.data, a.work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
truth: dict = {}
for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(par, set()).add(q)
orphans = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
bv = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet")
bt = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet")
cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
base = _chosen(bt, cfg)
for g in [cfg.get("gamma", 1.0)] + a.gammas:
    c = dict(cfg, gamma=g)
    chv = _chosen(bv, c)
    cht = _chosen(bt, c)
    print(f"gamma {g}: valB plain {_macro_hard(s1_b, truth, chv, orphans, 1):.5f} hard3 {_macro_hard(s1_b, truth, chv, orphans, 3):.5f} "
          f"hard5 {_macro_hard(s1_b, truth, chv, orphans, 5):.5f} | test pairs {cht.height} "
          f"(vs base -{base.join(cht, on=['q', 's1'], how='anti').height} +{cht.join(base, on=['q', 's1'], how='anti').height})", flush=True)
    if g in a.gammas:
        _write_outputs(a.work, a.out / f"gamma{g}", c, bt, cands, rec)
