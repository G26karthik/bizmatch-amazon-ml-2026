"""Deep dive 13: accepted pairs of fr16 by (name op x address op), France vs US vs India, per S1 rates, plus bracket
share. Classes over-represented in France relative to US are candidate French error pockets.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d13_op_classes.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a", "de", "du", "des", "la", "le", "les", "et"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); ct = dict(zip(rec["entity_id"], rec["country"]))
voc = {c: Counter(w for e, n in zip(rec["entity_id"], rec["business_name"]) if e.startswith("S1") and ct[e] == c for w in set(toks(n))) for c in ("France", "US", "India")}
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})


def name_op(q, s, c):
    A, B = set(toks(nm[q])) - LEG, set(toks(nm[s])) - LEG
    if A == B:
        return "same"
    add, drop = A - B, B - A
    if not A:
        return "only_legal"
    if len(add) == 1 and len(drop) == 1:
        a, d = next(iter(add)), next(iter(drop))
        if Levenshtein.distance(a, d) <= max(1, len(d) // 4):
            return "typo1"
        return "swap_vocab" if voc[c].get(a, 0) > 30 else "swap_nonword"
    if not drop and len(add) >= 1:
        return "add"
    if not add and len(drop) >= 1:
        return "drop"
    if len(add) == 1 and len(A) == 1:
        return "pseudo_or_handle"
    return "multi"


def addr_op(q, s):
    a, b = ad[q] or "", ad[s] or ""
    if not a.strip():
        return "empty"
    A, B = set(toks(a)), set(toks(b))
    ha = [t for t in A if t.isdigit()]; hb = [t for t in B if t.isdigit()]
    same_h = bool(set(ha) & set(hb)) if ha and hb else None
    j = len(A & B) / max(1, len(A | B))
    if A == B:
        return "exact"
    if j >= 0.6:
        return "near" + ("" if same_h is not False else "_hdiff")
    if same_h is False:
        return "house_diff"
    return "far"


rows = []
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    c = ct[s]
    for q in (m.split(",") if m else []):
        rows.append((c, name_op(q, s, c), addr_op(q, s), bool(re.search(r"[\[\(]", nm[q] or ""))))
x = pl.DataFrame(rows, schema=["country", "nop", "aop", "brk"], orient="row")
ns1 = rec.filter(pl.col("entity_id").str.starts_with("S1")).group_by("country").len()
t = x.group_by("country", "nop", "aop").agg(pl.len().alias("n"), pl.col("brk").mean().round(4).alias("brk")).join(ns1.rename({"len": "S1s"}), on="country")
t = t.with_columns((pl.col("n") / pl.col("S1s") * 1000).round(2).alias("per1kS1"))
p = t.pivot(on="country", index=["nop", "aop"], values=["per1kS1", "brk"]).fill_null(0)
p = p.with_columns((pl.col("per1kS1_France") / (pl.col("per1kS1_US") + 0.05)).round(2).alias("FR/US")).sort("per1kS1_France", descending=True)
print(p.head(60))
x.write_parquet("local_data/work/d13_classes.parquet")
