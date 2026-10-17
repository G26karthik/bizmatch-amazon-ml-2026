"""Merge France re-scores into cross-encoder score directories (environment-neutral delta).

For every France pair a model scores, the new score is
    sigmoid( logit(original score used so far) + logit(score of cleaned text) - logit(score of original text) ),
where the last two come from the same environment (laptop GPU or a Kaggle scoring job). Differences between
environments therefore cancel and only the effect of the France cleaning is transferred.

python scripts/france_ce_merge.py --ce /work/ce_wide_base2 --new /work/fr_ce/base2/new.parquet \
    --orig /work/fr_ce/base2/orig.parquet --out /work/ce_wide_base2_fr
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--ce", type=Path, required=True)
ap.add_argument("--new", type=Path, required=True)
ap.add_argument("--orig", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.is_symlink() or a.out.resolve() == a.ce.resolve():
    raise SystemExit(f"refusing to write into {a.out}: it is (a link to) the source directory {a.ce}")
a.out.mkdir(parents=True, exist_ok=True)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


old = pl.read_parquet(a.ce / "ce_scores_test.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
n = pl.read_parquet(a.new).rename({"pce": "pn"}).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
o = pl.read_parquet(a.orig).rename({"pce": "po"}).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
d = old.join(n, on=["q", "s1"], how="left").join(o, on=["q", "s1"], how="left")
hit = d["pn"].is_not_null().to_numpy() & d["po"].is_not_null().to_numpy()
z = logit(d["pce"].to_numpy())
z[hit] += logit(d["pn"].to_numpy()[hit]) - logit(d["po"].to_numpy()[hit])
new = 1 / (1 + np.exp(-z))
print(f"{a.ce.name}: {int(hit.sum())} France pairs updated of {d.height}; mean score {d['pce'].to_numpy()[hit].mean():.4f} -> "
      f"{new[hit].mean():.4f}; >0.5: {(d['pce'].to_numpy()[hit] > 0.5).mean():.4f} -> {(new[hit] > 0.5).mean():.4f}")
d.select("q", "s1").with_columns(pl.Series("pce", new.astype(np.float32))).write_parquet(a.out / "ce_scores_test.parquet")
import shutil  # noqa: E402
shutil.copy(a.ce / "ce_scores_val.parquet", a.out / "ce_scores_val.parquet")
