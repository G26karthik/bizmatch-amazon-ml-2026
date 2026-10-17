"""Deep dive 36: empty-address web-handle records (one token, or '.com'/'#'/'@' form): S1s of the country whose
concatenated name (all tokens / minus legal words) equals or contains the handle stem (>= 8 chars); unique -> candidate.
Train truth (all visible) precision; test unassigned counts (fr25).

uv run --no-project --with polars==1.44.2 python scripts/deep/d36_empty_handle.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
OUT = "local_data/work/empty_handle_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"[a-z0-9]+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def stem(n):
    t = [x for x in toks(n) if x not in ("com", "www", "net", "org", "in", "fr")]
    return "".join(t) if len(t) == 1 else None


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
    full, noleg = defaultdict(list), defaultdict(list)
    for e, n, c in s1.select("entity_id", "business_name", "country").iter_rows():
        T = toks(n)
        full[(c, "".join(T))].append(e)
        noleg[(c, "".join(x for x in T if x not in LEG))].append(e)
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
    q = q.filter(pl.col("business_name").fill_null("").str.contains(r"(?i)\.com\b|^#|^@|www\.") | ~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" "))
    if keep is not None:
        q = q.filter(pl.col("entity_id").is_in(keep))
    res = []
    for m, n, c in q.select("entity_id", "business_name", "country").iter_rows():
        st = stem(n)
        if not st or len(st) < 8:
            continue
        c1 = set(full.get((c, st), [])) | set(noleg.get((c, st), []))
        if len(c1) != 1:
            continue
        s = c1.pop()
        res.append((c, m, s, (truth.get(m) == s) if truth is not None else None))
    return pl.DataFrame(res, schema={"country": pl.String, "m": pl.String, "s1": pl.String, "right": pl.Boolean}, orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN precision:"); print(tr.group_by("country").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country"))
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(assigned.implode()))["entity_id"].implode())
print("TEST unassigned:"); print(te.group_by("country").len().sort("country"))
te.write_parquet(OUT.format("test"))
