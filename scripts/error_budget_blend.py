"""Error budget of a blend run on validation half B: gain from fixing each error type (oracle), with counts.

python scripts/error_budget_blend.py <tag> <work>   (reads val_report_<tag>.json, val_best_<tag>.parquet)
"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "src")
from ber.metric import entity_f05
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen
work = Path(sys.argv[2] if len(sys.argv) > 2 else "/work/ens2"); tag = sys.argv[1]
cfg = json.loads((work / f"val_report_{tag}.json").read_text())["best"]
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list(); sb = set(s1_b)
parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in parent.items(): truth.setdefault(p, set()).add(q)
best = pl.read_parquet(work / f"val_best_{tag}.parquet")
pred = {}
for q, s in _chosen(best, cfg).iter_rows(): pred.setdefault(s, set()).add(q)
sv = pl.read_parquet(work / "scores_val.parquet", columns=["q", "s1"])
cand = set(zip(sv["q"].to_list(), sv["s1"].to_list()))
bq = dict(zip(best["q"].to_list(), best["s1"].to_list()))
def score(pr): return float(np.mean([entity_f05(truth.get(s, ()), pr.get(s, ())) for s in s1_b]))
base = score(pred)
fo = {s: {q for q in v if q in parent} for s, v in pred.items()}
fw = {s: {q for q in v if q not in parent or parent[q] == s} for s, v in pred.items()}
# FN split: true pair is argmax of its record (decision dropped it) vs argmax is a different S1 (ranking error)
fn_dec = {s: set(pred.get(s, set())) | {q for q in truth.get(s, ()) if bq.get(q) == s} for s in sb}
fn_rank = {s: set(pred.get(s, set())) | {q for q in truth.get(s, ()) if (q, s) in cand and bq.get(q) != s} for s in sb}
lost = sum(1 for s in sb for q in truth.get(s, ()) if (q, s) not in cand)
fn_lost = {s: set(pred.get(s, set())) | {q for q in truth.get(s, ()) if (q, s) not in cand} for s in sb}
cnt = lambda f: sum(1 for s in sb for q in truth.get(s, ()) if f(q, s))
print(f"{tag} cfg={cfg} base={base:.5f}")
print(f"FP orphan  +{score(fo)-base:.5f} n={sum(1 for s,v in pred.items() if s in sb for q in v if q not in parent)}")
print(f"FP wrongS1 +{score(fw)-base:.5f} n={sum(1 for s,v in pred.items() if s in sb for q in v if q in parent and parent[q]!=s)}")
print(f"FN decision (argmax right, not chosen) +{score(fn_dec)-base:.5f} n={cnt(lambda q,s: bq.get(q)==s and q not in pred.get(s,set()))}")
print(f"FN ranking (argmax wrong S1)            +{score(fn_rank)-base:.5f} n={cnt(lambda q,s: (q,s) in cand and bq.get(q)!=s)}")
print(f"FN not in candidates                    +{score(fn_lost)-base:.5f} n={lost}")
