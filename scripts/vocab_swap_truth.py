"""US/India train truth: same-address one-word swaps by swapped-in word class — generic copy words, vocabulary words
(frequent in S1 names, not a near-spelling of the swapped-out word), typo-like (edit-similar to the swapped-out word),
other (rare/pseudo words). France analog: type words are S1 vocabulary words.

python scripts/vocab_swap_truth.py
"""
import sys
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
tr = ns["prep"]("/work/base/train/records.parquet")
d = ns["swaps"](tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
GEN = {"center", "services", "service", "partners", "centre"}
for c in ("US", "India"):
    voc = Counter(w for ws in tr.filter((pl.col("country") == c) & (pl.col("src") == 1))["nt"].to_list() for w in ws)
    x = d.filter(pl.col("country") == c)
    wout = []
    for a, b in zip(x["ntq"].to_list(), x["nts"].to_list()):
        wout.append(next(iter(set(b) - set(a))))
    cls = []
    for wi, wo in zip(x["w_in"].to_list(), wout):
        if wi in GEN:
            cls.append("generic")
        elif fuzz.ratio(wi, wo) >= 60:
            cls.append("typo_like")
        elif voc[wi] >= 50:
            cls.append("vocab")
        else:
            cls.append("rare")
    x = x.with_columns(pl.Series("cls", cls))
    print(c)
    print(x.group_by("arel", "cls").agg(pl.len(), pl.col("true").mean().round(4).alias("true_rate")).sort("arel", "cls"))
    v = x.filter((pl.col("arel") == "same") & (pl.col("cls") == "vocab"))
    print("  same-address vocab swaps, top in-words (true/false):", v.group_by("w_in").agg(pl.len(), pl.col("true").mean().round(3)).sort("len", descending=True).head(12).rows())
