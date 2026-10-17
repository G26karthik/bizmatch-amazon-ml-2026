"""E10 round 4 (half A only): structural variant - gamma strictness only for inclusion, the empty prediction valued with
the raw (un-gamma'd) probabilities; plus empirical check of the empty-prediction value."""
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402

t0 = time.time()
v, s1 = L.load()
B0 = L.score(L.decide(v, gamma=1.5), v, s1, "A")
for g in (1.5, 1.75, 2.0, 2.5):
    for m in (0.0, 0.5, 1.0):
        ch = L.decide(v, gamma=g, miss=m, mech=2, e0="raw")
        a = L.score(ch, v, s1, "A")
        print(f"g{g} empty=raw fn{m}: A plain {a[0]:.6f} ({a[0]-B0[0]:+.6f}) hard {a[1]:.6f} ({a[1]-B0[1]:+.6f}) pairs {ch.height}", flush=True)
# empirical: half-A S1s that production leaves empty although they have candidates: P(T=0) vs model estimates
b = v.filter(pl.col("prob") ** 1.5 >= 0.02)
ch = L.decide(v, gamma=1.5)
sA = s1.filter(pl.col("half_a")).join(ch.select("s1").unique(), on="s1", how="anti")
g = b.join(sA.select("s1", "T"), on="s1", how="inner").group_by("s1").agg(
    pl.col("prob").max().alias("pmax"), (1 - pl.col("prob")).product().alias("e_raw"), (1 - pl.col("prob") ** 1.5).product().alias("e_g"),
    pl.col("T").first(), pl.col("y").sum().alias("ty"))
for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    x = g.filter((pl.col("pmax") >= lo) & (pl.col("pmax") < hi))
    if x.height:
        print(f"empty-predicted A S1s, top p [{lo},{hi}): n={x.height} P(T=0)={(x['T'] == 0).mean():.3f} mean prod(1-p)={x['e_raw'].mean():.3f} "
              f"mean prod(1-p^1.5)={x['e_g'].mean():.3f} top candidate true={(x['ty'] > 0).mean():.3f}")
print("done", f"{time.time()-t0:.0f}s")
