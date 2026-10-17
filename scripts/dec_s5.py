"""E10 round 5 (half A only): stability filter - half A split into A1/A2 by S1 hash; a variant must gain in both parts
(plain and hard) before it may go to half B. Structural gamma variants (per source, per S1 size) + round-1 finalists."""
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402

t0 = time.time()
v, s1 = L.load()
sa = s1.filter(pl.col("half_a"))
part = (sa["s1"].hash(L.SEED + 23) % 2).to_numpy()
base = L.per_entity(L.decide(v, gamma=1.5), v, s1, "A")
assert (base["s1"] == sa["s1"].to_numpy()).all()

# calibration by source in the decision zone (half A)
z = v.filter(pl.col("s1").is_in(sa["s1"].implode()))
for src in (2, 3):
    for lo, hi in ((.3, .6), (.6, .8), (.8, .9), (.9, .97)):
        x = z.filter((pl.col("qsrc") == src) & (pl.col("prob") >= lo) & (pl.col("prob") < hi))
        print(f"A src{src} prob [{lo},{hi}): n={x.height} mean prob {x['prob'].mean():.4f} true {x['y'].mean():.4f} unmatched {x['orph'].mean():.4f}")


def run(name, best, **kw):
    r = L.per_entity(L.decide(best, **kw), best, s1, "A")
    s = []
    for m in ("plain", "hard"):
        d = r[m] - base[m]
        s.append(f"{m} A {d.mean():+.6f} A1 {d[part == 0].mean():+.6f} A2 {d[part == 1].mean():+.6f}")
    ok = all((r[m] - base[m])[part == k].mean() > 0 for m in ("plain", "hard") for k in (0, 1))
    print(f"{name:30s} {' | '.join(s)} {'STABLE' if ok else ''} [{time.time()-t0:.0f}s]", flush=True)


for name, kw in {"F1_g1.5_empty_e0.497": dict(gamma=1.5, miss=0.0, mech=2, e0=float(np.exp(-0.7))),
                 "F2_g1.75_miss1.0": dict(gamma=1.75, miss=1.0), "F3_g2.0_miss1.5": dict(gamma=2.0, miss=1.5),
                 "F4_g1.5_miss1.0": dict(gamma=1.5, miss=1.0), "gamma1.25": dict(gamma=1.25)}.items():
    run(name, v, **kw)
for g2 in (1.0, 1.25, 1.5):
    for g3 in (1.5, 1.75, 2.0, 2.5):
        if g2 == g3:
            continue
        run(f"src2_g{g2}_src3_g{g3}", v.with_columns(pl.when(pl.col("qsrc") == 2).then(g2).otherwise(g3).alias("gc")), gamma_col="gc")
vn = v.join(s1.select("s1", "n_best50"), on="s1", how="left").with_columns(pl.col("n_best50").fill_null(0))
for gs, gm, gl in ((1.25, 1.5, 1.75), (1.75, 1.5, 1.25), (1.5, 1.5, 2.0), (1.5, 1.5, 1.25), (1.25, 1.5, 1.5), (1.75, 1.5, 1.5)):
    run(f"size_g{gs}/{gm}/{gl}", vn.with_columns(pl.when(pl.col("n_best50") <= 1).then(gs).when(pl.col("n_best50") <= 3).then(gm)
                                                  .otherwise(gl).alias("gc")), gamma_col="gc")
print("done", f"{time.time()-t0:.0f}s")
