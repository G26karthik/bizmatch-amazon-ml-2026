"""Deep dive 35: alias records ('X d/b/a NAME', 'X trading as NAME', 'NAME formerly X', 't/a', 'aka'): the part on
either side of the alias marker whose nkey (tokens minus legal words) equals exactly one S1 of the country; address
relation (format-tolerant equal / other / empty). Train truth (all visible) precision; test unassigned counts (fr24).

uv run --no-project --with polars==1.44.2 python scripts/deep/d35_alias.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
ALIAS = re.compile(r"(?i)\s+(?:d\.?\s?b\.?\s?a\.?|trading as|formerly:?|aka|a\.k\.a\.|t/a)\s+")
OUT = "local_data/work/alias_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def nkey(s):
    return frozenset(set(toks(s)) - LEG)


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
    by = defaultdict(list); addr = {}
    for e, n, a, c in s1.select("entity_id", "business_name", "business_address", "country").iter_rows():
        by[(c, nkey(n))].append(e); addr[e] = a
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & pl.col("business_name").fill_null("").str.contains(ALIAS.pattern))
    if keep is not None:
        q = q.filter(pl.col("entity_id").is_in(keep))
    res = []
    for m, n, a, c in q.select("entity_id", "business_name", "business_address", "country").iter_rows():
        parts = [p for p in ALIAS.split(n) if p.strip()]
        hits = set()
        for p in parts:
            k = nkey(p)
            if k and len(by.get((c, k), [])) == 1:
                hits.add(by[(c, k)][0])
        if len(hits) != 1:
            continue
        s = hits.pop()
        A, B = set(toks(a)), set(toks(addr[s]))
        da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
        arel = "empty" if not A else ("same" if da == db and len(A & B) / len(A | B) >= 0.35 else "other")
        res.append((c, m, s, arel, (truth.get(m) == s) if truth is not None else None))
    return pl.DataFrame(res, schema=["country", "m", "s1", "arel", "right"], orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN precision (alias part names exactly one S1):")
print(tr.group_by("country", "arel").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country", "arel"))
g = pl.read_csv("output/v28q_fr24/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(assigned.implode()))["entity_id"].implode())
print("TEST unassigned:")
print(te.group_by("country", "arel").len().sort("country", "arel"))
te.write_parquet(OUT.format("test"))
