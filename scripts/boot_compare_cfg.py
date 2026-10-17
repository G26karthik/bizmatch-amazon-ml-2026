"""Paired bootstrap over half-B S1 entities between two saved blend runs at ONE fixed decision config (default production:
efopt, gamma 1.5), on both the plain metric and the hard metric (unmatched-record false matches counted three times).

python scripts/boot_compare_cfg.py /work/ens2 run28_7q run28_8q [--gamma 1.5]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

work, ta, tb = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
gamma = float(sys.argv[sys.argv.index("--gamma") + 1]) if "--gamma" in sys.argv else 1.5
cfg = {"mode": "efopt", "miss": 0.0, "gamma": gamma}
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
truth = {}
for q, p in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
res = {}
for tag in (ta, tb):
    ch = _chosen(pl.read_parquet(work / f"val_best_{tag}.parquet"), cfg)
    plain, hard = {}, {}
    for q, s in ch.iter_rows():
        plain.setdefault(s, set()).add(q)
        hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    res[tag] = {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1_b]) for m, pm in (("plain", plain), ("hard", hard))}
    print(tag, cfg, {m: round(float(v.mean()), 6) for m, v in res[tag].items()}, flush=True)
rng = np.random.default_rng(0)
for m in ("plain", "hard"):
    d = res[tb][m] - res[ta][m]
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    print(f"{m}: diff={d.mean():+.6f} 95%CI=[{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(diff<=0)={np.mean(bs <= 0):.3f} better={np.sum(d > 0)} worse={np.sum(d < 0)}")
