"""Error budget of the v4 model on validation half B (run4): how much macro F0.5 each error type costs.

For each error type we "fix" only that type (oracle) and recompute macro F0.5; the gain is that type's cost.
Types: FP from unmatched (orphan) records; FP from records that belong to another S1 (wrong S1);
FN where the true pair was a candidate but not chosen; FN where the true pair was pruned/never retrieved.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

work = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/run4")
rep_name = sys.argv[2] if len(sys.argv) > 2 else "val_report_s2.json"
best_name = sys.argv[3] if len(sys.argv) > 3 else "val_best_s2.parquet"
cfg = json.loads((work / rep_name).read_text())["best"]
lab = labels(Path("/work/data"), work).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
sb = set(s1_b)
parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth: dict = {}
for q, p in parent.items():
    truth.setdefault(p, set()).add(q)
best = pl.read_parquet(work / best_name)
ch = _chosen(best, cfg)
pred: dict = {}
for q, s in ch.iter_rows():
    pred.setdefault(s, set()).add(q)
cand = set()
for p in sorted((work / "train" / "pruned").glob("*.parquet")):
    d = pl.read_parquet(p, columns=["q", "s1"]).filter(pl.col("s1").is_in(pl.Series(s1_b).implode()))
    cand.update(zip(d["q"].to_list(), d["s1"].to_list()))


def score(pr):
    return float(np.mean([entity_f05(truth.get(s, ()), pr.get(s, ())) for s in s1_b]))


base = score(pred)
fix_orphan_fp = {s: {q for q in v if q in parent} for s, v in pred.items()}
fix_wrong_fp = {s: {q for q in v if q not in parent or parent[q] == s} for s, v in pred.items()}
fix_fn_cand = {s: set(pred.get(s, set())) | {q for q in truth.get(s, ()) if (q, s) in cand} for s in sb}
fix_all_cand = {s: {q for q in truth.get(s, ()) if (q, s) in cand} for s in sb}
n_fp_orphan = sum(1 for s, v in pred.items() if s in sb for q in v if q not in parent)
n_fp_wrong = sum(1 for s, v in pred.items() if s in sb for q in v if q in parent and parent[q] != s)
n_fn_cand = sum(1 for s in sb for q in truth.get(s, ()) if (q, s) in cand and q not in pred.get(s, set()))
n_fn_lost = sum(1 for s in sb for q in truth.get(s, ()) if (q, s) not in cand)
print(f"valB entities {len(s1_b)}  base macro F0.5 {base:.5f}")
print(f"fix FP from orphan records   (+{score(fix_orphan_fp) - base:.5f})  count {n_fp_orphan}")
print(f"fix FP to the wrong S1       (+{score(fix_wrong_fp) - base:.5f})  count {n_fp_wrong}")
print(f"fix FN among candidates      (+{score(fix_fn_cand) - base:.5f})  count {n_fn_cand}")
print(f"perfect choice from candidates = {score(fix_all_cand):.5f}  (pruned/unretrieved true pairs: {n_fn_lost})")
sing = [s for s in s1_b if not truth.get(s)]
print(f"singleton S1s {len(sing)}: wrongly given matches {sum(1 for s in sing if pred.get(s))}")
