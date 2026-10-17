"""Deep dive 24: any S2/S3 record whose strict address key (house number + street tokens, words in >1% of the
country's S1 addresses dropped) equals that of exactly one S1: P(record is a copy of that S1) on train truth (all S1
visible = test-like), by name similarity (rapidfuzz token_set_ratio bins) and whether the record is unassigned in test.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d24_strict_addr.py
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
OUT = "local_data/work/strict_addr_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)]).filter(pl.col("country") != "France")
    res = []
    for ctry in ("US", "India"):
        r = rec.filter(pl.col("country") == ctry)
        s1 = r.filter(pl.col("entity_id").str.starts_with("S1"))
        df = Counter(t for a in s1["business_address"].to_list() for t in set(toks(a)))
        common = {t for t, c in df.items() if c > 0.01 * s1.height}
        akey = lambda a: frozenset(t for t in toks(a) if t not in common)  # noqa: E731
        by = defaultdict(list)
        for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
            k = akey(a)
            if any(t.isdigit() for t in k) and len(k) >= 2:
                by[k].append((e, n))
        q = r.filter(~pl.col("entity_id").str.starts_with("S1"))
        if keep is not None:
            q = q.filter(pl.col("entity_id").is_in(keep))
        elif split == "train":
            q = q.sample(fraction=0.35, seed=1)
        for m, n, a in q.select("entity_id", "business_name", "business_address").iter_rows():
            c = by.get(akey(a), [])
            if len(c) != 1:
                continue
            s, sn = c[0]
            sim = fuzz.token_set_ratio(" ".join(toks(n)), " ".join(toks(sn)))
            res.append((ctry, m, s, sim, (truth.get(m) == s) if truth is not None else None, (truth.get(m) is None) if truth is not None else None))
    x = pl.DataFrame(res, schema=["country", "m", "s1", "sim", "right", "noise"], orient="row")
    return x.with_columns(pl.col("sim").cut([30, 50, 60, 70, 80, 90]).alias("simb"))


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN: records at the strict address of exactly one S1")
print(tr.group_by("country", "simb").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right"), pl.col("noise").mean().round(4).alias("P_noise")).sort("country", "simb"))
g = pl.read_csv("output/v28q_fr20/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
un = allq.filter(~pl.col("entity_id").is_in(list(assigned)))["entity_id"]
te = run("test", keep=un.implode())
print("TEST unassigned records at the strict address of exactly one S1")
print(te.group_by("country", "simb").len().sort("country", "simb"))
te.write_parquet(OUT.format("test")); tr.write_parquet(OUT.format("train"))
