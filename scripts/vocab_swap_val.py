"""Validation check: accepted (chosen) val pairs that are same-address one-word swaps to a vocabulary word (frequent S1
name word, not generic copy word, not a near-spelling). US/India truth: such swaps are ~0% true for real words. Reports
count, precision and the half A / B effect of removing them (standard and test-like).

python scripts/vocab_swap_val.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
work, tag = Path("/work/run28"), "run28_7q"
rec = ns["prep"](str(work / "train" / "records.parquet"))
GEN = {"center", "services", "service", "partners", "centre"}
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))
S = {h: s1v.filter(pl.col("half_a") == (h == "A"))["rid"].to_list() for h in "AB"}
parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in parent.items():
    truth.setdefault(p, set()).add(q)
gt = pl.read_parquet("/work/data/train_pairs.parquet")
hidE = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19)).select(pl.col("entity_id").alias("s1"))
H = set(gt.join(hidE, on="s1", how="semi").select(pl.col("m").alias("entity_id")).join(rec.select("entity_id", "rid"), on="entity_id")["rid"].to_list())
cfg = json.loads((work / f"val_report_{tag}.json").read_text())["best"]
vb = pl.read_parquet(work / f"val_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
d = ns["swaps"](rec).filter(pl.col("arel") == "same").select("q", "s1", "country", "ntq", "nts", "w_in")
d = d.join(ch, on=["q", "s1"])
voc = {c: Counter(w for ws in rec.filter((pl.col("country") == c) & (pl.col("src") == 1))["nt"].to_list() for w in ws) for c in ("US", "India")}
keep = []
for c, a, b, wi in zip(d["country"].to_list(), d["ntq"].to_list(), d["nts"].to_list(), d["w_in"].to_list()):
    wo = next(iter(set(b) - set(a)))
    keep.append(wi not in GEN and fuzz.ratio(wi, wo) < 60 and voc[c][wi] >= 50 and len(wi) >= 3 and not wi.isdigit())
fl = d.filter(pl.Series(keep)).select("q", "s1", "country", "w_in")
fl = fl.with_columns(pl.Series("true", [parent.get(q) == s for q, s in zip(fl["q"].to_list(), fl["s1"].to_list())]))
print("accepted val same-address vocab swaps:", fl.height, "precision", fl["true"].mean(), fl.group_by("country").agg(pl.len(), pl.col("true").mean()).rows())
print("top words:", fl.group_by("w_in").agg(pl.len(), pl.col("true").mean().round(2)).sort("len", descending=True).head(12).rows())


def macro(chs, half, drop_h=False):
    pm = {}
    for q, s in chs.iter_rows():
        if drop_h and q in H:
            continue
        pm.setdefault(s, set()).add(q)
    return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S[half]]))


new = ch.join(fl.select("q", "s1"), on=["q", "s1"], how="anti")
for h in "AB":
    for t in (False, True):
        b0, b1 = macro(ch, h, t), macro(new, h, t)
        print(f"half {h} {'test-like' if t else 'standard'}: {b0:.6f} -> {b1:.6f} ({b1 - b0:+.6f})")
