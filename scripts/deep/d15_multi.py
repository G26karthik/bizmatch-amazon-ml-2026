"""Deep dive 15: accepted 'multi' name-op pairs (>= 2 token changes, legal forms ignored) in fr16. Bracket share
conditional on the record carrying a legal-form token; examples for France.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d15_multi.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
LEGAL = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp", "pc", "plc",
         "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "ei"}
LEG = LEGAL | {"the", "and", "of", "l", "c", "p", "d", "s", "a", "de", "du", "des", "la", "le", "les", "et"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); ct = dict(zip(rec["entity_id"], rec["country"]))
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
rows, ex = [], []
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        tq = toks(nm[q]); A, B = set(tq) - LEG, set(toks(nm[s])) - LEG
        add, drop = A - B, B - A
        kind = "same" if A == B else ("multi" if (len(add) + len(drop) >= 3 or (len(add) >= 2) or (len(drop) >= 2)) else "one")
        has_leg = bool(set(tq) & LEGAL)
        brk = bool(re.search(r"[\[\(\{]", nm[q] or ""))
        rows.append((ct[s], kind, has_leg, brk, len(add), len(drop)))
        if ct[s] == "France" and kind == "multi" and len(ex) < 4000:
            ex.append((nm[q], ad[q], nm[s], ad[s], sorted(add), sorted(drop)))
x = pl.DataFrame(rows, schema=["country", "kind", "has_leg", "brk", "n_add", "n_drop"], orient="row")
print(x.group_by("country", "kind", "has_leg").agg(pl.len(), pl.col("brk").mean().round(4)).sort("country", "kind", "has_leg"))
print(x.filter(pl.col("kind") == "multi").group_by("country", pl.col("n_add").clip(0, 3), pl.col("n_drop").clip(0, 3)).len().sort("country", "len", descending=[False, True]).head(30))
import random
random.seed(5)
for e in random.sample(ex, 30):
    print(f"  {e[0]!r:42s} | {e[1]!s:50.50s} -> {e[2]!r:36s} | add {e[4]} drop {e[5]}")
