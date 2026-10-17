"""E10 round 3 (half A only): the gain of `miss` comes from the empty-prediction factor exp(-miss). Wider grid with the two
effects separated (fn-only miss m_fn, empty factor e0 = exp(-m_e)), and decomposition of the changed decisions."""
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
base_ch = L.decide(v, gamma=1.5)


def decomp(ch, half="A"):
    """Changed S1 decisions vs production: empty->non-empty and additions/removals, by outcome of changed pairs."""
    ss = s1.filter(pl.col("half_a") == (half == "A")).select("s1")
    a = base_ch.join(ss, on="s1", how="semi").select("q", "s1")
    b = ch.join(ss, on="s1", how="semi").select("q", "s1")
    add = b.join(a, on=["q", "s1"], how="anti").join(v.select("q", "parent", "orph"), on="q", how="left")
    rem = a.join(b, on=["q", "s1"], how="anti").join(v.select("q", "parent", "orph"), on="q", how="left")
    was_empty = add.join(a.select("s1").unique(), on="s1", how="anti")

    def cnt(d):
        tp = int((d["parent"] == d["s1"]).fill_null(False).sum())
        orph = int(d["orph"].sum())
        return f"{d.height} (true {tp}, orphan {orph}, other-S1 copy {d.height - tp - orph})"
    return f"added {cnt(add)} [to empty S1s: {cnt(was_empty)}] removed {cnt(rem)}"


def run(name, best, show=False, **kw):
    ch = L.decide(best, **kw)
    a = L.score(ch, best, s1, "A")
    res[name] = {"A_plain": a[0], "A_hard": a[1], "pairs": ch.height, "kw": kw}
    print(f"{name:34s} A plain {a[0]:.6f} ({a[0]-B0[0]:+.6f}) hard {a[1]:.6f} ({a[1]-B0[1]:+.6f}) pairs {ch.height} [{time.time()-t0:.0f}s]", flush=True)
    if show:
        print("    ", decomp(ch), flush=True)
    return a


B0 = (0.0, 0.0)
B0 = run("base_g1.5", v, gamma=1.5)
# empty factor only
for g in (1.5, 1.75, 2.0, 2.5):
    for me in (0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0):
        run(f"g{g}_emptyonly_me{me}", v, show=(g == 1.5), gamma=g, miss=0.0, mech=2, e0=float(np.exp(-me)))
# fn-only miss + empty factor
for g in (1.5, 1.75, 2.0):
    for mf in (0.1, 0.2, 0.4):
        for me in (0.4, 1.0, 2.0):
            run(f"g{g}_fn{mf}_me{me}", v, gamma=g, miss=mf, mech=2, e0=float(np.exp(-me)))
# full miss, larger values
for g in (1.5, 1.75, 2.0, 2.5):
    for m in (1.5, 2.0, 3.0):
        for pois in (0, 1):
            run(f"g{g}_miss{m}_pois{pois}", v, show=(g == 1.75 and pois == 0), gamma=g, miss=m, pois=pois)
for name in ("g1.5_miss1.0", "g1.75_miss1.0", "g1.75_miss0.8"):
    g, m = float(name.split("_")[0][1:]), float(name.split("miss")[1])
    run(name + "_pois0", v, show=True, gamma=g, miss=m)
json.dump(res, open("/work/agent_dec/s3_results.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
