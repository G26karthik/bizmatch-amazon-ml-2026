"""Deep dive 28: records (non-empty address) whose name key (tokens minus legal forms) equals that of exactly one S1 of
the country, by address relation (token Jaccard bins). Train truth (all visible) P(copy of that S1); test (fr21)
unassigned counts per bin.

uv run --no-project --with polars==1.44.2 python scripts/deep/d28_samename_far.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
    by = defaultdict(list)
    for e, n, a, c in s1.select("entity_id", "business_name", "business_address", "country").iter_rows():
        by[(c, frozenset(set(toks(n)) - LEG))].append((e, set(toks(a))))
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
    q = q.filter(pl.col("entity_id").is_in(keep)) if keep is not None else q.sample(fraction=0.3, seed=4)
    res = []
    for m, n, a, c in q.select("entity_id", "business_name", "business_address", "country").iter_rows():
        k = frozenset(set(toks(n)) - LEG)
        if not k:
            continue
        cs = by.get((c, k), [])
        if len(cs) != 1:
            continue
        s, SA = cs[0]; A = set(toks(a))
        j = len(A & SA) / max(1, len(A | SA))
        res.append((c, m, s, j, (truth.get(m) == s) if truth is not None else None, (truth.get(m) is None) if truth is not None else None))
    return pl.DataFrame(res, schema=["country", "m", "s1", "j", "right", "noise"], orient="row").with_columns(pl.col("j").cut([0.1, 0.2, 0.3, 0.45, 0.6]).alias("jb"))


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print(tr.group_by("country", "jb").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right"), pl.col("noise").mean().round(4).alias("P_noise")).sort("country", "jb"))
g = pl.read_csv("output/v28q_fr21/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(list(assigned)))["entity_id"].implode())
print(te.group_by("country", "jb").len().sort("country", "jb"))
te.write_parquet("local_data/work/samename_far_test.parquet")
