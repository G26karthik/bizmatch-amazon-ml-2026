"""Deep dive 14: single-token pseudo-brand / handle records accepted in fr16. Split handle (derived from the S1 name:
concatenation similarity) vs pseudo (unrelated token); per country: per-1k-S1 rates, address class, how many S1s the
same token is accepted at, multi-tenant S1 address, and examples.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d14_pseudo.py
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a", "de", "du", "des", "la", "le", "les", "et",
       "com", "www", "net", "org", "in", "fr", "one", "group", "groupe", "global", "hq", "online"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def akey(s):
    return " ".join(sorted(toks(s)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); ct = dict(zip(rec["entity_id"], rec["country"]))
s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
akc = Counter((c, akey(a)) for c, a in zip(s1["country"], s1["business_address"]))
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
rows = []
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        A = set(toks(nm[q])) - LEG; B = set(toks(nm[s])) - LEG
        if len(A) != 1 or A <= B:
            continue
        t = next(iter(A))
        conc = "".join(w for w in toks(nm[s]))
        handle = fuzz.partial_ratio(t, conc) >= 80 or fuzz.ratio(t, conc) >= 70
        a, b = set(toks(ad[q] or "")), set(toks(ad[s] or ""))
        arel = "empty" if not a else ("exact" if a == b else ("near" if len(a & b) / len(a | b) >= 0.6 else "far"))
        rows.append((ct[s], s, q, t, "handle" if handle else "pseudo", arel, akc[(ct[s], akey(ad[s]))] >= 2))
x = pl.DataFrame(rows, schema=["country", "s1", "q", "tok", "kind", "arel", "multi_tenant"], orient="row")
ns1 = s1.group_by("country").len().rename({"len": "S1s"})
tokS1 = x.group_by("country", "tok").agg(pl.col("s1").n_unique().alias("tok_S1s"))
x = x.join(tokS1, on=["country", "tok"])
t = x.group_by("country", "kind", "arel").agg(pl.len().alias("n"), pl.col("multi_tenant").mean().round(3).alias("mt"),
                                               (pl.col("tok_S1s") >= 2).mean().round(3).alias("tok_reused")).join(ns1, on="country")
print(t.with_columns((pl.col("n") / pl.col("S1s") * 1000).round(2).alias("per1k")).sort("kind", "arel", "country"))
for c in ("France", "US"):
    ex = x.filter((pl.col("country") == c) & (pl.col("kind") == "pseudo")).sample(12, seed=4)
    print("\n", c, "pseudo examples")
    for r in ex.iter_rows(named=True):
        print(f"   {nm[r['q']]!r:40s} | {ad[r['q']]}  ->  {nm[r['s1']]!r} | {ad[r['s1']]}  [tok at {r['tok_S1s']} S1s, mt={r['multi_tenant']}]")
print("\nmost reused pseudo tokens France:", x.filter((pl.col("country") == "France") & (pl.col("kind") == "pseudo")).group_by("tok").len().sort("len", descending=True).head(15).rows())
print("most reused pseudo tokens US:", x.filter((pl.col("country") == "US") & (pl.col("kind") == "pseudo")).group_by("tok").len().sort("len", descending=True).head(15).rows())
x.write_parquet("local_data/work/d14_pseudo.parquet")
