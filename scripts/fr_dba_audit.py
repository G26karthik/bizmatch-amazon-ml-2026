"""Accepted France multi-token-difference pairs (diff2..diff6 in fr_name_ops_audit.parquet): strip a pseudo-brand DBA /
AKA / T-A prefix or suffix ("Wexflux aka QRX Lycee SARL", "QRX Lycee SARL t/a Wexflux") and re-classify the remainder
against the S1 name key: identical (copy: one DBA op), one-word swap to a vocabulary word (DBA + sibling swap = two ops ->
decoy), non-word swap, other. Bracket share of the swapped word as a copy marker.

python scripts/fr_dba_audit.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
LEG = LEGF | {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie", "service"}
SEP = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|a\s*/\s*k\s*/\s*a|t\s*/\s*a|dba|aka|trading as|doing business as)\b", re.I)


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name", "src", "country"])
nm = rec["business_name"].to_list()
voc1 = Counter(w for n in rec.filter((pl.col("country") == "France") & (pl.col("src") == 1))["business_name"].to_list()
               for w in set(re.findall(r"[a-z0-9]+", norm(n))))
x = pl.read_parquet("/work/lead/fr_name_ops_audit.parquet").filter(pl.col("pat").str.starts_with("diff"))
rows = []
for q, s in zip(x["q"].to_list(), x["s1"].to_list()):
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
    parts = SEP.split(norm(nm[q]))
    if len(parts) < 2:
        cls, br = "no_dba", False
    else:
        best = None
        for p in parts:  # the part closest to the S1 name is the original name
            A = set(re.findall(r"[a-z0-9]+", p)) - LEG
            if best is None or len(A ^ B) < len(best[0] ^ B):
                best = (A, p)
        A, p = best
        if A == B:
            cls = "dba_identical"
        elif len(A - B) == 1 and len(B - A) == 1:
            w = next(iter(A - B))
            cls = "dba_swap_generic" if w in GEN else ("dba_swap_vocab" if voc1.get(w, 0) > 30 else "dba_swap_nonword")
        elif B < A or A < B:
            cls = "dba_add_drop"
        else:
            cls = "dba_other"
        w = next(iter(A - B)) if len(A - B) == 1 else None
        br = bool(w) and bool(re.search(r"[\[\(]\s*" + re.escape(w), p))
    rows.append((cls, br))
x = x.with_columns(pl.Series("dcls", [r[0] for r in rows]), pl.Series("dbr", [r[1] for r in rows]))
print(x.group_by("dcls").agg(pl.len().alias("n"), pl.col("dbr").mean().round(4).alias("brk"), pl.col("single").mean().round(3).alias("single"),
      (pl.col("arel") == "same_house").mean().round(3).alias("same_h")).sort("n", descending=True))
for c in ("dba_swap_vocab", "dba_other", "no_dba"):
    z = x.filter(pl.col("dcls") == c)
    print(f"\nexamples {c} ({z.height})")
    for q, s in z.sample(n=min(10, z.height), seed=9).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:42} | {nm[q]!r}")
x.write_parquet("/work/lead/fr_dba_audit.parquet")
