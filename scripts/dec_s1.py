"""E09/E10 decoder variants on cached validation predictions (run28_7q), tuned on half A only.
Prints half-A (and, for bookkeeping only, nothing from half B) plain / hard macro F0.5 for every variant.

python scripts/dec_s1.py   (on the worker, after dec_s0.py --build)
"""
import json
import sys
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402

t0 = time.time()
v, s1 = L.load()
hA = dict(s1.select("s1", "half_a").iter_rows())
v = v.with_columns(pl.col("s1").replace_strict(hA, default=None, return_dtype=pl.Boolean).alias("hA"))
res = {}


def run(name, best, **kw):
    ch = L.decide(best, **kw)
    a = L.score(ch, best, s1, "A")
    res[name] = {"A_plain": a[0], "A_hard": a[1], "pairs": ch.height}
    print(f"{name:42s} A plain {a[0]:.6f} hard {a[1]:.6f} pairs {ch.height}  [{time.time()-t0:.0f}s]", flush=True)
    return a


base = run("base_g1.5", v, gamma=1.5)
for g in (1.0, 1.25, 1.75, 2.0, 2.5):
    run(f"gamma{g}", v, gamma=g)

# ---- calibration diagnostics in the competitor zone (half A)
z = v.filter(pl.col("hA").fill_null(False))
for lo, hi in ((0, .01), (.01, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    x = z.filter((pl.col("c_max") >= lo) & (pl.col("c_max") < hi) & (pl.col("prob") > 0.5))
    print(f"A prob>0.5, c_max [{lo},{hi}): n={x.height} mean prob {x['prob'].mean():.4f} true rate {x['y'].mean():.4f}")
for lo, hi in ((.5, .8), (.8, .9), (.9, .97), (.97, .995), (.995, 1.01)):
    x = z.filter((pl.col("c_max") >= .01) & (pl.col("prob") >= lo) & (pl.col("prob") < hi))
    print(f"A c_max>=0.01, prob [{lo},{hi}): n={x.height} mean prob {x['prob'].mean():.4f} true rate {x['y'].mean():.4f}")

# ---- (b) miss: constant
for pois in (0, 1):
    for m in (0.02, 0.05, 0.1, 0.2, 0.4):
        run(f"miss{m}_pois{pois}", v, gamma=1.5, miss=m, pois=pois)

# ---- (b) miss model: E[true copies outside the argmax list | other-candidate mass, confident count], half A table
sA = s1.filter(pl.col("half_a")).with_columns((pl.col("T") - pl.col("T_best")).alias("mt"))
edges = [0.01, 0.1, 0.3, 0.6]


def cells(df):
    return df.with_columns(pl.col("om_sum").cut(edges, labels=[str(i) for i in range(5)]).cast(pl.Utf8).alias("c_om"),
                           pl.col("n_best50").clip(0, 3).cast(pl.Utf8).alias("c_nb"))


tab = cells(sA).group_by("c_om", "c_nb").agg(pl.col("mt").mean().alias("mhat"), pl.len().alias("n"))
print(tab.sort("c_om", "c_nb"))
s1m = cells(s1).join(tab.select("c_om", "c_nb", "mhat"), on=["c_om", "c_nb"], how="left").with_columns(pl.col("mhat").fill_null(0.06))
vm = v.join(s1m.select("s1", "mhat"), on="s1", how="left").with_columns(pl.col("mhat").fill_null(0.06))
for lam in (0.5, 1.0, 1.5):
    vm = vm.with_columns((pl.col("mhat") * lam).alias("mm"))
    for pois in (0, 1):
        run(f"missmodel_x{lam}_pois{pois}", vm, gamma=1.5, miss="mm", pois=pois)

# ---- (a) record competition: renormalisation with a 'none' option (other candidates' p3 as their probability)
vr = v.with_columns(
    (pl.col("prob") / pl.max_horizontal(pl.lit(1.0), pl.col("prob") + pl.col("c_sum"))).alias("p_norm"),
    (pl.col("prob") * (1 - pl.col("c_max"))).alias("p_excl"),
    (pl.col("prob") / (pl.col("prob") + pl.col("c_sum"))).alias("p_soft"))
for col in ("p_norm", "p_excl", "p_soft"):
    run(f"{col}_g1.5", vr, pcol=col, gamma=1.5)
    run(f"{col}_g1.0", vr, pcol=col, gamma=1.0)


# ---- (a) record competition: logistic calibrator in the competitor zone (c_max >= 0.01), cross-fitted on half A
def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def F(df):
    lp, lc, ls = lg(df["prob"].to_numpy()), lg(df["c_max"].to_numpy()), lg(df["p3self"].to_numpy())
    lsum = np.log(np.clip(df["c_sum"].to_numpy(), 1e-6, None))
    return np.c_[lp, lc, ls, lsum, np.log1p(df["n_other"].to_numpy()), lp * lc]


def calib(v, zmin=0.01):
    zone = (pl.col("c_max") >= zmin) & pl.col("hA").is_not_null()
    zv = v.with_row_index("_i").filter(zone)
    isA = zv["hA"].to_numpy()
    X, y = F(zv), zv["y"].to_numpy().astype(int)
    fold = (zv["s1"].hash(L.SEED + 11) % 2).to_numpy()
    out = np.empty(zv.height)
    for k in (0, 1):
        tr = isA & (fold != k)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        out[isA & (fold == k)] = m.predict_proba(X[isA & (fold == k)])[:, 1]
    mall = LogisticRegression(C=1.0, max_iter=2000).fit(X[isA], y[isA])
    out[~isA] = mall.predict_proba(X[~isA])[:, 1]
    print(f"calibrator zone>= {zmin}: n={zv.height} A={isA.sum()} coef {mall.coef_.round(3).tolist()} {mall.intercept_.round(3).tolist()}", flush=True)
    newp = v["prob"].to_numpy().astype(np.float64).copy()
    newp[zv["_i"].to_numpy()] = out
    return v.with_columns(pl.Series("p_cal", newp)), mall


for zmin in (0.01, 0.001):
    vc, _ = calib(v, zmin)
    for g in (1.0, 1.5, 2.0):
        run(f"calib_z{zmin}_g{g}", vc, pcol="p_cal", gamma=g)

# ---- (c) gamma per bucket: stricter gamma in the competitor zone
for gz in (2.0, 3.0, 4.0):
    vg = v.with_columns(pl.when(pl.col("c_max") >= 0.01).then(gz).otherwise(1.5).alias("gcol"))
    run(f"gamma_zone{gz}", vg, gamma_col="gcol")
for g2, g3 in ((1.25, 1.75), (1.75, 1.25), (1.5, 2.0), (2.0, 1.5)):
    vg = v.with_columns(pl.when(pl.col("qsrc") == 2).then(g2).otherwise(g3).alias("gcol"))
    run(f"gamma_src2_{g2}_src3_{g3}", vg, gamma_col="gcol")
run("caps_g1.5", v, gamma=1.5, caps=True)
json.dump(res, open("/work/agent_dec/s1_results.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
