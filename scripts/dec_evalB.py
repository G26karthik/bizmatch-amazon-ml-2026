"""Finalists chosen on half A -> ONE evaluation on half B: paired bootstrap over half-B S1 entities vs production
(efopt gamma 1.5, miss 0): plain (official), hard (unmatched-record false matches x3, as scripts/eval_judge.py) and
hardE (hard + a singleton S1 whose prediction contains an unmatched record scores -2 instead of 0, i.e. such losses x3:
the test's ~2x decoy density acts on empty-vs-one decisions too, which `hard` cannot see)."""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402

v, s1 = L.load()
FIN = {
    "F1_g1.5_empty_e0.497": dict(gamma=1.5, miss=0.0, mech=2, e0=float(np.exp(-0.7))),
    "F2_g1.75_miss1.0": dict(gamma=1.75, miss=1.0),
    "F3_g2.0_miss1.5": dict(gamma=2.0, miss=1.5),
    "F4_g1.5_miss1.0": dict(gamma=1.5, miss=1.0),
}


def arrays(ch, half):
    r = L.per_entity(ch, v, s1, half)
    ss = s1.filter(pl.col("half_a") == (half == "A")).select("s1", "T")
    assert (ss["s1"].to_numpy() == r["s1"]).all()
    c = ch.select("q", "s1").join(v.select("q", "orph"), on="q", how="left").group_by("s1").agg(pl.col("orph").sum().alias("fo"))
    fo = ss.join(c, on="s1", how="left")["fo"].fill_null(0).to_numpy()
    he = np.where((ss["T"].to_numpy() == 0) & (fo > 0), -2.0, r["hard"])
    return {"plain": r["plain"], "hard": r["hard"], "hardE": he}


base = L.decide(v, gamma=1.5)
out = {}
for half in ("A", "B"):
    a0 = arrays(base, half)
    print(f"== half {half}: production plain {a0['plain'].mean():.6f} hard {a0['hard'].mean():.6f} hardE {a0['hardE'].mean():.6f}", flush=True)
    ss = s1.filter(pl.col("half_a") == (half == "A")).select("s1")
    bset = base.join(ss, on="s1", how="semi").select("q", "s1")
    for name, kw in FIN.items():
        ch = L.decide(v, **kw)
        a1 = arrays(ch, half)
        cset = ch.join(ss, on="s1", how="semi").select("q", "s1")
        add = cset.join(bset, on=["q", "s1"], how="anti").join(v.select("q", "parent", "orph"), on="q", how="left")
        rem = bset.join(cset, on=["q", "s1"], how="anti").join(v.select("q", "parent", "orph"), on="q", how="left")
        aff = pl.concat([add.select("s1"), rem.select("s1")]).unique().height
        print(f"-- {name} {kw}")
        print(f"   affected S1s {aff}; added {add.height} (true {int((add['parent'] == add['s1']).fill_null(False).sum())}, "
              f"unmatched {int(add['orph'].sum())}); removed {rem.height} (true {int((rem['parent'] == rem['s1']).fill_null(False).sum())}, "
              f"unmatched {int(rem['orph'].sum())})")
        out.setdefault(name, {})[half] = {}
        for m in ("plain", "hard", "hardE"):
            s = L.boot(a0[m], a1[m])
            out[name][half][m] = s
            print(f"   {m:5s} {s}", flush=True)
json.dump(out, open("/work/agent_dec/evalB_results.json", "w"), indent=1)
