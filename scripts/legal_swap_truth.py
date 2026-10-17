"""US/India train truth: legal-form agreement between a one-word-swap record and its S1, for true vs false records by
address relation; checks the France rule 'type-word swap + legal-form change => decoy' (France copies keep the legal
form 97.6%, decoys 51%).

python scripts/legal_swap_truth.py
"""
import re
import sys
import unicodedata

import polars as pl

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
LF = {"inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company", "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def legal(s):
    return frozenset(set(re.findall(r"[a-z]+", norm(s).replace(".", ""))) & LF)


tr = ns["prep"]("/work/base/train/records.parquet")
d = ns["swaps"](tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
nm = tr["business_name"].to_list()
rel = []
for q, s in zip(d["q"].to_list(), d["s1"].to_list()):
    a, b = legal(nm[q]), legal(nm[s])
    rel.append("same" if a == b else ("rec_none" if not a else ("s1_none" if not b else "diff")))
d = d.with_columns(pl.Series("legal_rel", rel))
for c in ("US", "India"):
    x = d.filter(pl.col("country") == c)
    print(c)
    print(x.group_by("arel", "true", "legal_rel").len().pivot(on="legal_rel", index=["arel", "true"], values="len").fill_null(0).sort("arel", "true"))
    print(" true rate by legal_rel at same address:", x.filter(pl.col("arel") == "same").group_by("legal_rel").agg(pl.len(), pl.col("true").mean().round(4)).rows())
