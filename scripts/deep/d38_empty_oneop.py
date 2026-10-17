"""Deep dive 38: empty-address records whose name is ONE word-operation away from exactly one S1 of the country
(nkey = tokens minus legal words; ops: record adds a word, drops a word, swaps a word, typo in one word). Train truth
(all S1 visible = test-like) precision by op; test unassigned counts (fr25), with production best pair status.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d38_empty_oneop.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
OUT = "local_data/work/empty_oneop_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def key(ts):
    return " ".join(sorted(ts))


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
    exact, minus = defaultdict(list), defaultdict(list)
    for e, n, c in s1.select("entity_id", "business_name", "country").iter_rows():
        T = set(toks(n)) - LEG
        if not T:
            continue
        exact[(c, key(T))].append(e)
        if len(T) >= 2:
            for t in T:
                minus[(c, key(T - {t}))].append((e, t))
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
    if keep is not None:
        q = q.filter(pl.col("entity_id").is_in(keep))
    res = []
    for m, n, c in q.select("entity_id", "business_name", "country").iter_rows():
        R = set(toks(n)) - LEG
        if not R or exact.get((c, key(R))):
            continue                        # exact-name cases handled elsewhere (unique) or ties
        cand = {}
        if len(R) >= 2:
            for r in R:                     # record ADDED word r
                for e in exact.get((c, key(R - {r})), ()):
                    cand.setdefault(e, "add")
        for e, t in minus.get((c, key(R)), ()):   # record DROPPED word t
            cand.setdefault(e, "drop")
        for r in R:                         # record SWAPPED / TYPO'D word r -> t
            for e, t in minus.get((c, key(R - {r})), ()):
                if t != r:
                    cand.setdefault(e, "typo" if Levenshtein.distance(r, t) <= max(1, min(len(r), len(t)) // 4) else "swap")
        if len(cand) != 1:
            continue
        e, op = next(iter(cand.items()))
        res.append((c, m, e, op, len(R), (truth.get(m) == e) if truth is not None else None))
    return pl.DataFrame(res, schema={"country": pl.String, "m": pl.String, "s1": pl.String, "op": pl.String, "ntok": pl.Int64, "right": pl.Boolean}, orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN (all visible) precision, empty-address record one op from exactly one S1:")
print(tr.group_by("country", "op").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country", "op"))
print(tr.group_by("op", pl.col("ntok").clip(1, 4)).agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("op", "ntok"))
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(assigned.implode()))["entity_id"].implode())
print("TEST unassigned:")
print(te.group_by("country", "op").len().sort("country", "op"))
te.write_parquet(OUT.format("test")); tr.write_parquet(OUT.format("train"))
