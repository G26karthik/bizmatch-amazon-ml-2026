"""E01 side check 3: validation precision of exactly the class fr10 removes when the swapped-in word is a non-word:
same-address (addr_n token multiset) one-token name swap, legal form kept, swapped-in word absent from the country's S1
names (df 0), NOT a near-spelling by fr10's test (ratio >= 70, 4-char prefix, in-order abbreviation). US / India labels.
Also the ratio profile of the France restore candidates.

python scripts/e01_nonword_val2.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_e01")
from ber.pipeline import labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402
from e01_gap_lib import LEGALW  # noqa: E402

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", norm(s)) if t not in LEGALW]


def legal(s):
    return sorted(t for t in re.findall(r"[a-z0-9]+", norm(s)) if t in LEGALW)


def abbrev(x, y):
    if not x or not y or x[0] != y[0] or len(x) >= len(y):
        return False
    it = iter(y)
    return all(ch in it for ch in x)


def near(wi, wo):
    return fuzz.ratio(wi, wo) >= 70 or wi.startswith(wo[:4]) or wo.startswith(wi[:4]) or abbrev(wi, wo) or abbrev(wo, wi)


def swap(nq, ns):
    a, b = Counter(toks(nq)), Counter(toks(ns))
    i, o = list((a - b).elements()), list((b - a).elements())
    return (i[0], o[0]) if len(i) == 1 and len(o) == 1 else (None, None)


work = Path("/work/run28")
lab = labels(Path("/work/data"), work).select(pl.col("rid").cast(pl.UInt32).alias("q"), "parent")
vb = pl.read_parquet(work / "val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
tr = pl.read_parquet(work / "train" / "records.parquet", columns=["src", "country", "business_name", "addr_n"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
dfc = {c: Counter(t for n in tr.filter((pl.col("src") == 1) & (pl.col("country") == c))["business_name"].to_list() for t in set(toks(n))) for c in ("US", "India")}
v = vb.join(tr.select(pl.col("rid").alias("q"), "country", pl.col("business_name").alias("nq"), pl.col("addr_n").alias("aq")), on="q") \
      .join(tr.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns"), pl.col("addr_n").alias("as_")), on="s1") \
      .join(lab, on="q", how="left").join(ch.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left") \
      .with_columns(pl.col("acc").fill_null(False), (pl.col("parent").cast(pl.UInt32) == pl.col("s1")).fill_null(False).alias("y"))
v = v.filter((pl.col("aq") != "") & pl.Series([sorted(a.split()) == sorted(b.split()) for a, b in zip(v["aq"].to_list(), v["as_"].to_list())]))
rows = []
for c, nq, ns in zip(v["country"].to_list(), v["nq"].to_list(), v["ns"].to_list()):
    wi, wo = swap(nq, ns)
    if wi is None:
        rows.append((None, None, None, None, None))
        continue
    rows.append((wi, wo, dfc[c].get(wi, 0), near(wi, wo), legal(nq) == legal(ns)))
v = v.with_columns(pl.Series("w_in", [r[0] for r in rows]), pl.Series("w_out", [r[1] for r in rows]), pl.Series("df_in", [r[2] for r in rows]),
                   pl.Series("near", [r[3] for r in rows]), pl.Series("legal_kept", [r[4] for r in rows])).filter(pl.col("w_in").is_not_null())
v = v.with_columns(pl.when(pl.col("df_in") == 0).then(pl.lit("nonword")).when(pl.col("df_in") < 20).then(pl.lit("rare<20")).otherwise(pl.lit("word")).alias("cls"))
print("=== validation same-address one-token swaps: precision by word class x fr10 near-spelling test (legal kept only)")
print(v.filter(pl.col("legal_kept")).group_by("country", "cls", "near").agg(pl.len(), pl.col("y").mean().alias("prec"), pl.col("acc").mean().alias("acc_rate"))
       .sort("country", "cls", "near"))
print(v.filter(pl.col("legal_kept") & (pl.col("cls") == "nonword") & ~pl.col("near")).select("country", "nq", "ns", "w_in", "w_out", "y").sample(20, seed=3))
print(v.filter(pl.col("legal_kept") & (pl.col("cls") == "nonword") & ~pl.col("near") & ~pl.col("y")).select("country", "nq", "ns", "w_in", "w_out", "prob").head(15))
