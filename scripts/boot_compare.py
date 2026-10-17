"""Paired bootstrap over half-B S1 entities: per-entity hard F0.5 difference between two saved blend runs.

python scripts/boot_compare.py /work/ens2 tagA:val_report_tagA.json tagB:val_report_tagB.json
"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "src")
from ber.metric import entity_f05
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen
work = Path(sys.argv[1]); pairs = [a.split(":") for a in sys.argv[2:]]  # tag:reportname
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
truth = {}
for q, p in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
per = []
for tag, rep in pairs:
    cfg = json.loads((work / rep).read_text())["blend"]["hard"]["cfg"]
    ch = _chosen(pl.read_parquet(work / f"val_best_{tag}.parquet"), cfg)
    pm = {}
    for q, s in ch.iter_rows():
        pm.setdefault(s, set()).add(q)
        if q in orph: pm[s] |= {("d", q, 1), ("d", q, 2)}
    per.append(np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1_b]))
    print(tag, cfg, f"hard_valB={per[-1].mean():.5f}", flush=True)
d = per[1] - per[0]
rng = np.random.default_rng(0); n = len(d)
bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
print(f"n={n} diff={d.mean():+.5f}  95%CI=[{np.quantile(bs,.025):+.5f},{np.quantile(bs,.975):+.5f}]  P(diff<=0)={np.mean(bs<=0):.3f}  changed={np.sum(d!=0)} better={np.sum(d>0)} worse={np.sum(d<0)}")
