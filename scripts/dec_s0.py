"""Sanity: rebuild cache, reproduce production half-B numbers three ways, and diagnostics for decoder ideas."""
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_dec")
import dec_lib as L  # noqa: E402
from ber.metric import entity_f05  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

t0 = time.time()
if "--build" in sys.argv:
    L.build()
v, s1 = L.load()
print("loaded", v.shape, s1.shape, f"{time.time()-t0:.0f}s", flush=True)
best = v.select("q", "s1", "prob")
cfg = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
ch0 = _chosen(best, cfg)
# eval_judge-style loop
truth = {}
for q, p in v.filter(pl.col("parent").is_not_null()).select("q", "parent").iter_rows():
    truth.setdefault(p, set()).add(q)
orph = set(v.filter(pl.col("orph"))["q"].to_list())
SB = s1.filter(~pl.col("half_a"))["s1"].to_list()
plain, hard = {}, {}
for q, s in ch0.select("q", "s1").iter_rows():
    plain.setdefault(s, set()).add(q)
    hard.setdefault(s, set()).add(q)
    if q in orph:
        hard[s] |= {("d", q, 1), ("d", q, 2)}
fb = {m: np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in SB]) for m, pm in (("plain", plain), ("hard", hard))}
print("loop metric (truth from val records only):", fb, flush=True)
print("vectorised, production decoder: B", L.score(ch0, v, s1, "B"), "A", L.score(ch0, v, s1, "A"))
ch1 = L.decide(v, gamma=1.5)
print("vectorised, dec_lib decoder:    B", L.score(ch1, v, s1, "B"), "same pairs:",
      ch1.join(ch0, on=["q", "s1"], how="anti").height == 0 and ch0.join(ch1, on=["q", "s1"], how="anti").height == 0, flush=True)
print("T check: truth copies of val S1 missing from val records:", int(s1["T"].sum()) - v.filter(pl.col("parent").is_not_null() & pl.col("parent").is_in(s1["s1"].implode())).height)

# diagnostics
c = ch1.join(v, on=["q", "s1"], how="left")
print("\nchosen pairs", c.height, "precision", c["y"].mean())
for lo, hi in ((0, .01), (.01, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    x = c.filter((pl.col("c_max") >= lo) & (pl.col("c_max") < hi))
    print(f"  chosen with competitor p3 max in [{lo},{hi}): n={x.height} prec={x['y'].mean() if x.height else float('nan'):.4f}"
          f" orph={x['orph'].mean() if x.height else float('nan'):.4f}")
nc = v.join(ch1.select("q"), on="q", how="anti")
print("unchosen records", nc.height, "whose 2nd candidate is the true parent:",
      nc.filter(pl.col("s2nd") == pl.col("parent")).height,
      "of those with 2nd p3>0.5:", nc.filter((pl.col("s2nd") == pl.col("parent")) & (pl.col("c_max") > 0.5)).height,
      "any 2nd p3>0.5:", nc.filter(pl.col("c_max") > 0.5).height)
wrong = v.filter(pl.col("parent").is_not_null() & (pl.col("parent") != pl.col("s1")))
print("records whose argmax is a wrong S1:", wrong.height, "true parent is a candidate:", wrong.filter(pl.col("p3true").is_not_null()).height,
      "p3true>0.5:", wrong.filter(pl.col("p3true") > 0.5).height)
cs = c.group_by(["s1", "qsrc"]).len()
print("chosen over cap: S2>5", cs.filter((pl.col("qsrc") == 2) & (pl.col("len") > 5)).height,
      "S3>6", cs.filter((pl.col("qsrc") == 3) & (pl.col("len") > 6)).height)
bs = v.filter(pl.col("prob") ** 1.5 >= 0.02).group_by(["s1", "qsrc"]).len()
print("candidate lists over cap: S2>5", bs.filter((pl.col("qsrc") == 2) & (pl.col("len") > 5)).height,
      "S3>6", bs.filter((pl.col("qsrc") == 3) & (pl.col("len") > 6)).height)
tc = v.filter(pl.col("parent").is_not_null()).group_by(["parent", "qsrc"]).len()
print("truth max copies per src:", tc.group_by("qsrc").agg(pl.col("len").max()).to_dicts())
# miss structure on half A
sA = s1.filter(pl.col("half_a")).with_columns((pl.col("T") - pl.col("T_best")).alias("miss_true"))
print("\nhalf A: mean T", sA["T"].mean(), "mean missed (true copies not argmax-assigned)", sA["miss_true"].mean(),
      "P(T=0)", (sA["T"] == 0).mean())
print(sA.with_columns(pl.col("n_best50").clip(0, 6).alias("nb")).group_by("nb").agg(
    pl.len(), pl.col("miss_true").mean(), (pl.col("T") == 0).mean().alias("p_single")).sort("nb"))
print(sA.with_columns(pl.col("om_sum").cut([0.01, 0.1, 0.3, 0.6, 1.0]).alias("om")).group_by("om").agg(
    pl.len(), pl.col("miss_true").mean()).sort("om"))
# per-entity errors on half B by category
r = L.per_entity(ch1, v, s1, "B")
bad = (r["plain"] < 1).sum()
print("\nhalf B entities with F<1:", int(bad), "of", len(r["plain"]), "loss", float((1 - r["plain"]).sum()))
print(f"done {time.time()-t0:.0f}s")
