"""Where a US/India-trained probability model disagrees with the French decisions of a submission, which side is right?
Reads local_data/llm/resid_test.parquet (q, s1 rids of the concatenated test sources; prob = production, pn = new).
Sets: REMOVE = accepted French pairs with pn < --lo; ADD = French pairs whose record is unassigned with pn > --hi.
For each set: name-difference pattern, bracket share of the new word (French copies ~4.2%, decoys ~0.9%), same-house
share, and a full-cap test for ADD (records at S1s whose source already holds 5 S2 / 6 S3 accepted matches: copies
cannot be there).

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/fr_resid_check.py --base output/v28q_fr16 [--lo 0.2 --hi 0.9]
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, default=Path("output/v28q_fr16"))
ap.add_argument("--newp", type=Path, default=Path("local_data/llm/resid_test.parquet"))
ap.add_argument("--lo", type=float, default=0.2)
ap.add_argument("--hi", type=float, default=0.9)
a = ap.parse_args()
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie", "service"}


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); eid = rec["entity_id"].to_list(); srcl = rec["src"].to_list()
fr_rows = rec.filter(pl.col("country") == "France")
voc = Counter(w for n in fr_rows.filter(pl.col("src") == 1)["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", asc(n))))
rid_of = {e: i for i, e in enumerate(eid)}
g = pl.read_csv(a.base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = {}
cnt = Counter()
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for e in (m.split(",") if m else []):
        acc[rid_of[e]] = rid_of[s]; cnt[(rid_of[s], srcl[rid_of[e]])] += 1
x = pl.read_parquet(a.newp).join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1").filter(pl.col("country") == "France")
print("French uncertain pairs:", x.height)


def pattern(q, s):
    A = set(re.findall(r"[a-z0-9]+", asc(nm[q]))) - LEG
    B = set(re.findall(r"[a-z0-9]+", asc(nm[s]))) - LEG
    new = None
    if A == B:
        p = "identical_key"
    elif len(A - B) == 1 and len(B - A) == 1:
        new = next(iter(A - B)); p = "SWAP_" + ("generic" if new in GEN else ("vocab" if voc.get(new, 0) > 30 else "nonword"))
    elif B < A and len(A - B) == 1:
        new = next(iter(A - B)); p = "ADD_" + ("generic" if new in GEN else ("vocab" if voc.get(new, 0) > 30 else "nonword"))
    elif A < B:
        p = "DROP"
    else:
        p = "other"
    br = bool(new) and bool(re.search(r"[\[\(]\s*" + re.escape(new), asc(nm[q])))
    h1 = re.search(r"\b(\d{1,5})\b", asc(ad[s])); h2 = re.search(r"\b(\d{1,5})\b", asc(ad[q]))
    same_h = bool(h1 and h2 and h1.group(1) == h2.group(1))
    return p, br, same_h


rows = []
for q, s, p0, pn in x.select("q", "s1", "prob", "pn").iter_rows():
    cur = acc.get(q)
    if cur == s and pn < a.lo:
        kind = "REMOVE"
    elif cur is None and pn > a.hi:
        kind = "ADD"
    else:
        continue
    pat, br, sh = pattern(q, s)
    full = cnt[(s, srcl[q])] >= (5 if srcl[q] == 2 else 6)
    rows.append((kind, pat, br, sh, full, p0, pn))
y = pl.DataFrame(rows, schema=["kind", "pat", "bracket", "same_house", "at_cap", "p0", "pn"], orient="row")
print(y.group_by("kind", "pat").agg(pl.len().alias("n"), pl.col("bracket").mean().round(4).alias("brk"), pl.col("same_house").mean().round(3).alias("same_h"),
      pl.col("at_cap").sum().alias("n_at_cap"), pl.col("p0").mean().round(3).alias("p0"), pl.col("pn").mean().round(3).alias("pn")).sort("kind", "n", descending=[False, True]))
print("totals:", y.group_by("kind").len().rows())
