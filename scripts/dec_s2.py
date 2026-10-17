"""E10 round 2 (half A only): which part of `miss` helps, miss x gamma grid, miss model variants, per-bucket gamma."""
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402

t0 = time.time()
v, s1 = L.load()
res = {}


def run(name, best, **kw):
    ch = L.decide(best, **kw)
    a = L.score(ch, best, s1, "A")
    res[name] = {"A_plain": a[0], "A_hard": a[1], "pairs": ch.height, "kw": {k: x for k, x in kw.items() if not isinstance(x, pl.DataFrame)}}
    print(f"{name:40s} A plain {a[0]:.6f} ({a[0]-B0[0]:+.6f}) hard {a[1]:.6f} ({a[1]-B0[1]:+.6f}) pairs {ch.height}  [{time.time()-t0:.0f}s]", flush=True)
    return a


B0 = (0.0, 0.0)
B0 = run("base_g1.5", v, gamma=1.5)
# mechanism split
for m in (0.1, 0.2, 0.4):
    run(f"miss{m}_emptyonly", v, gamma=1.5, miss=m, mech=1)
    run(f"miss{m}_fnonly", v, gamma=1.5, miss=m, mech=2)
# miss x gamma grid (deterministic + poisson)
for g in (1.5, 1.75, 2.0, 2.5):
    for m in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        for pois in (0, 1):
            run(f"g{g}_miss{m}_pois{pois}", v, gamma=g, miss=m, pois=pois)

# miss model, finer: per-S1 expected missed copies from half-A cells, several scalings, plus additive constant
sA = s1.filter(pl.col("half_a")).with_columns((pl.col("T") - pl.col("T_best")).alias("mt"))
edges = [0.01, 0.1, 0.3, 0.6]


def cells(df):
    return df.with_columns(pl.col("om_sum").cut(edges, labels=[str(i) for i in range(5)]).cast(pl.Utf8).alias("c_om"),
                           pl.col("n_best50").clip(0, 3).cast(pl.Utf8).alias("c_nb"))


tab = cells(sA).group_by("c_om", "c_nb").agg(pl.col("mt").mean().alias("mhat"), pl.len().alias("n"))
s1m = cells(s1).join(tab.select("c_om", "c_nb", "mhat"), on=["c_om", "c_nb"], how="left").with_columns(pl.col("mhat").fill_null(0.06))
vm = v.join(s1m.select("s1", "mhat"), on="s1", how="left").with_columns(pl.col("mhat").fill_null(0.06))
for g in (1.5, 2.0):
    for lam, add in ((2.0, 0), (3.0, 0), (4.0, 0), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3), (2.0, 0.2)):
        vm = vm.with_columns((pl.col("mhat") * lam + add).alias("mm"))
        for pois in (0, 1):
            run(f"g{g}_mm_x{lam}+{add}_pois{pois}", vm, gamma=g, miss="mm", pois=pois)
# om_sum directly as miss (other-candidate mass = expected true copies whose argmax is elsewhere under p3)
for lam, add in ((1.0, 0.0), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3), (2.0, 0.2)):
    vo = v.join(s1.select("s1", "om_sum"), on="s1", how="left").with_columns((pl.col("om_sum").fill_null(0) * lam + add).alias("mo"))
    for pois in (0, 1):
        run(f"g1.5_om_x{lam}+{add}_pois{pois}", vo, gamma=1.5, miss="mo", pois=pois)
# gamma by confidence structure of the S1: S1s with no confident candidate (empty vs one weak record)
vn = v.join(s1.select("s1", "n_best50"), on="s1", how="left").with_columns(pl.col("n_best50").fill_null(0))
for g0 in (1.0, 1.25, 2.0, 3.0):
    run(f"gamma_nb0_{g0}", vn.with_columns(pl.when(pl.col("n_best50") == 0).then(g0).otherwise(1.5).alias("gcol")), gamma_col="gcol")
for g1 in (1.25, 1.75, 2.0):
    run(f"gamma_nbpos_{g1}", vn.with_columns(pl.when(pl.col("n_best50") == 0).then(1.5).otherwise(g1).alias("gcol")), gamma_col="gcol")
json.dump(res, open("/work/agent_dec/s2_results.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
