"""Deep dive 41: pseudo-brand tokens as per-S1 aliases. For one-token S2/S3 records whose token (len >= 6, alphabetic,
not related to any S1 name) is carried by OTHER records assigned to exactly one S1 (train: truth copies; test: fr44
assignments) and by no other S1's records: P(record belongs to that S1). Train truth (all visible); test counts.

uv run --no-project --with polars==1.44.2 python scripts/deep/d41_pseudo_alias.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)


def tok(n):
    s = unicodedata.normalize("NFKD", n or "").casefold()
    t = re.findall(r"[a-z]+", "".join(ch for ch in s if not unicodedata.combining(ch)))
    return t[0] if len(t) == 1 and len(t[0]) >= 6 and " " not in (n or "").strip() else None


def run(split, owner, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (2, 3)])
    ct = dict(zip(rec["entity_id"].to_list(), rec["country"].to_list()))
    toks = {e: tok(n) for e, n in zip(rec["entity_id"].to_list(), rec["business_name"].to_list())}
    by = defaultdict(set)                      # (country, token) -> S1s owning a record with that token
    for e, s in owner.items():
        t = toks.get(e)
        if t:
            by[(ct[e], t)].add(s)
    res = []
    cand = keep if keep is not None else [e for e, t in toks.items() if t]
    for e in cand:
        t = toks.get(e)
        if not t:
            continue
        own = by.get((ct[e], t), set())
        if truth is not None:                 # leave-one-out: remove the record's own contribution
            own = set(own)
            s_own = truth.get(e)
            if s_own is not None:
                others = [x for x in rec_by_tok[(ct[e], t)] if x != e and truth.get(x) == s_own]
                if not others:
                    own.discard(s_own)
        if len(own) != 1:
            continue
        s = next(iter(own))
        res.append((ct[e], e, s, (truth.get(e) == s) if truth is not None else None))
    return pl.DataFrame(res, schema={"country": pl.String, "m": pl.String, "s1": pl.String, "right": pl.Boolean}, orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
truth = dict(zip(tp["m"].to_list(), tp["s1"].to_list()))
trrec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (2, 3)])
rec_by_tok = defaultdict(list)
for e, n, c in trrec.select("entity_id", "business_name", "country").iter_rows():
    t = tok(n)
    if t:
        rec_by_tok[(c, t)].append(e)
tr = run("train", truth, truth=truth)
print("TRAIN (leave-one-out) pseudo token owned by exactly one S1 via other records:")
print(tr.group_by("country").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country"))
g = pl.read_csv("output/v28q_fr44/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
owner = dict(zip(a["matched_entity_ids"].to_list(), a["source1_entity_id"].to_list()))
te_all = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])["entity_id"].to_list()
un = [e for e in te_all if e not in owner]
te = run("test", owner, keep=un)
print("TEST unassigned resolvable:"); print(te.group_by("country").len().sort("country"))
te.with_columns(pl.lit("alias").alias("kind")).write_parquet("local_data/work/pseudo_alias_test.parquet")
