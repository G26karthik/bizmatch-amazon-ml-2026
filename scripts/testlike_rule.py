"""Test-like decision rule.

Evidence (2026-09-26): test's extra unmatched records are decoys, not copies of businesses missing from S1 (share of
true-copy-flagged records test/train = 0.81-0.82, the decoy prediction; absent copies would give ~1.08). Our 19%
hiding simulation creates exactly such absent-S1 copies ("H" records: raw parent hidden), so the standard validation
over-penalises matching copy-like records and pushes the decision rule too strict (the leaderboard confirmed: a
stricter rule lost 0.0003).

Test-like validation: drop H records from the universe (they do not exist in test) and count false matches from
remaining orphans k=2 times (test has ~2x the decoy density of raw training data). Choose separate strictness
exponents for empty-address and address records on test-like half A; report half B (plain and k2) with a paired
bootstrap against the production rule; optionally write test outputs with the chosen rule.

python scripts/testlike_rule.py --work /work/run24 --tag run24_both [--out /work/out_v27_tl]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, decide_pairs, hidden_s1, labels  # noqa: E402
from ber.stage2 import _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/work/data"))
ap.add_argument("--base", type=Path, default=Path("/work/base"))
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--ge", default="0.6,0.8,1.0,1.25,1.5", help="strictness grid for empty-address records")
ap.add_argument("--ga", default="0.8,1.0,1.25,1.5,2.0", help="strictness grid for records with an address")
ap.add_argument("--rule_tag", help="write testlike_rule_<rule_tag>.json instead of testlike_rule_<tag>.json")
ap.add_argument("--out", type=Path)
ap.add_argument("--ref", help="reference tag with its own testlike_rule_<ref>.json: paired bootstrap on test-like half B")
ap.add_argument("--ref_work", type=Path, help="work dir of the reference tag (default --work)")
a = ap.parse_args()

cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
g_prod = float(cfg.get("gamma", 1.0))
lab = labels(a.data, a.base).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
rec = pl.read_parquet(a.base / "train" / "records.parquet", columns=["entity_id", "src", "addr_empty"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
# H records: raw ground-truth parent is a hidden S1 (they only exist because of the simulation)
gt = pl.read_parquet(a.data / "train_pairs.parquet")
hid = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19)).select(pl.col("entity_id").alias("s1"))
H = gt.join(hid, on="s1", how="semi").select(pl.col("m").alias("entity_id")).join(rec.select("entity_id", "rid"), on="entity_id")["rid"]
Hset = set(H.to_list())
print(f"H records (raw parent hidden): {len(Hset)}")
truth: dict = {}
for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(par, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list()) - Hset
s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val"))
S1A, S1B = s1.filter(pl.col("half_a"))["rid"].to_list(), s1.filter(~pl.col("half_a"))["rid"].to_list()
best = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
best = best.filter(~pl.col("q").is_in(H.implode())).join(rec.select(pl.col("rid").alias("q"), "addr_empty"), on="q", how="left")


def choose(b: pl.DataFrame, ge: float, ga: float) -> pl.DataFrame:
    g = pl.when(pl.col("addr_empty").fill_null(False)).then(ge).otherwise(ga)
    return decide_pairs(b.with_columns(pl.col("prob").pow(g).alias("prob")).select("q", "s1", "prob"), 0.0)


def per_entity(chosen: pl.DataFrame, s1s: list, k: int) -> np.ndarray:
    pm: dict = {}
    for q, s in chosen.iter_rows():
        pm.setdefault(s, set()).add(q)
        if q in orph:
            for j in range(1, k):
                pm[s].add(("decoy", q, j))
    return np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1s])


grid = [(ge, ga) for ge in map(float, a.ge.split(",")) for ga in map(float, a.ga.split(","))]
resA = []
for ge, ga in grid:
    c = choose(best, ge, ga)
    resA.append((per_entity(c, S1A, a.k).mean(), ge, ga))
resA.sort(reverse=True)
print("top rules on test-like half A (k=%d):" % a.k, [(round(s, 5), ge, ga) for s, ge, ga in resA[:5]])
_, ge, ga = resA[0]
prod = choose(best, g_prod, g_prod)
new = choose(best, ge, ga)
for name, c in (("production", prod), (f"test-like ({ge},{ga})", new)):
    print(f"{name}: test-like half B plain {per_entity(c, S1B, 1).mean():.5f}  k{a.k} {per_entity(c, S1B, a.k).mean():.5f}")
d = per_entity(new, S1B, a.k) - per_entity(prod, S1B, a.k)
rng = np.random.default_rng(0)
bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
print(f"test-like half B k{a.k}: diff {d.mean():+.5f} CI [{np.quantile(bs, .025):+.5f},{np.quantile(bs, .975):+.5f}] P<=0 {np.mean(bs <= 0):.3f} "
      f"better {int((d > 0).sum())} worse {int((d < 0).sum())}")
# standard (hiding) validation for reference: the looser rule must not collapse there
bstd = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)) \
         .join(rec.select(pl.col("rid").alias("q"), "addr_empty"), on="q", how="left")
orph_all = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
orph_bak, orph = orph, orph_all
for name, (x, y) in (("production", (g_prod, g_prod)), ("test-like rule", (ge, ga))):
    c = choose(bstd, x, y)
    print(f"standard half B ({name}): plain {per_entity(c, S1B, 1).mean():.5f} hard3 {per_entity(c, S1B, 3).mean():.5f}")
orph = orph_bak
json.dump({"ge": ge, "ga": ga, "k": a.k, "gamma_prod": g_prod}, open(a.work / f"testlike_rule_{a.rule_tag or a.tag}.json", "w"))
if a.ref:
    rw = a.ref_work or a.work
    rr = json.loads((rw / f"testlike_rule_{a.ref}.json").read_text())
    bref = pl.read_parquet(rw / f"val_best_{a.ref}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    bref = bref.filter(~pl.col("q").is_in(H.implode())).join(rec.select(pl.col("rid").alias("q"), "addr_empty"), on="q", how="left")
    cref = choose(bref, rr["ge"], rr["ga"])
    for k in (1, a.k):
        e_new, e_ref = per_entity(new, S1B, k), per_entity(cref, S1B, k)
        d = e_new - e_ref
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"vs {a.ref} ({rr['ge']},{rr['ga']}): test-like half B k{k} {e_ref.mean():.5f} -> {e_new.mean():.5f} diff {d.mean():+.5f} "
              f"CI [{np.quantile(bs, .025):+.5f},{np.quantile(bs, .975):+.5f}] P<=0 {np.mean(bs <= 0):.3f} "
              f"better {int((d > 0).sum())} worse {int((d < 0).sum())}")
if a.out is not None:
    tb = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    trec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "addr_empty"]).with_row_index("rid")
    tb = tb.join(trec.select(pl.col("rid").cast(pl.UInt32).alias("q"), "addr_empty"), on="q", how="left")
    ct = choose(tb, ge, ga)
    cp = choose(tb, g_prod, g_prod)
    print(f"test: production {cp.height} pairs -> test-like rule {ct.height} (+{ct.join(cp, on=['q', 's1'], how='anti').height} "
          f"-{cp.join(ct, on=['q', 's1'], how='anti').height})")
    cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
    _write_outputs(a.work, a.out, {"mode": "thr", "thr": 0.5}, ct.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob")),
                   cands, trec.select("rid", "entity_id", "src"))
