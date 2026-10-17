"""Deep dive 37: per class (name relation x address relation) -> train TRUE copies per 1k S1 (all visible) vs test
ACCEPTED pairs per 1k S1 (fr25), US and India. Copies come from the same generator in train and test, so
test_accepted - train_copies (per S1) lower-bounds accepted decoys (test has ~2x decoys) in a class.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d37_class_excess.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def nrel(a, b):
    A, B = set(toks(a)) - LEG, set(toks(b)) - LEG
    if not A:
        return "only_legal"
    if A == B:
        return "same"
    add, drop = A - B, B - A
    if len(add) == 1 and len(drop) == 1:
        x, y = next(iter(add)), next(iter(drop))
        return "typo" if Levenshtein.distance(x, y) <= 2 else "swap"
    if not drop:
        return "add"
    if not add:
        return "drop"
    return "multi"


def arel(a, b):
    A, B = set(toks(a)), set(toks(b))
    if not A:
        return "empty"
    da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
    j = len(A & B) / max(1, len(A | B))
    if da == db and j >= 0.35:
        return "same"
    if j >= 0.35:
        return "near_hdiff"
    return "far"


def rates(split, pairs, frac=1.0):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); ct = dict(zip(rec["entity_id"], rec["country"]))
    n1 = rec.filter(pl.col("entity_id").str.starts_with("S1")).group_by("country").len()
    rows = [(ct[s], nrel(nm[m], nm[s]), arel(ad[m], ad[s])) for s, m in pairs if ct[s] != "France"]
    x = pl.DataFrame(rows, schema=["country", "nrel", "arel"], orient="row")
    return x.group_by("country", "nrel", "arel").len().join(n1.rename({"len": "S1s"}), on="country").with_columns(
        (pl.col("len") / pl.col("S1s") / frac * 1000).round(2).alias(f"per1k_{split}")).drop("len", "S1s")


tp = pl.read_parquet("local_data/train_pairs.parquet").sample(fraction=0.5, seed=1)
tr = rates("train", zip(tp["s1"].to_list(), tp["m"].to_list()), frac=0.5)
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
te = rates("test", zip(a["source1_entity_id"].to_list(), a["matched_entity_ids"].to_list()))
x = tr.join(te, on=["country", "nrel", "arel"], how="full", coalesce=True).fill_null(0).with_columns(
    (pl.col("per1k_test") - pl.col("per1k_train")).round(2).alias("excess"), (pl.col("per1k_test") / (pl.col("per1k_train") + 0.01)).round(3).alias("ratio"))
print(x.sort("country", "excess", descending=[False, True]))
x.write_parquet("local_data/work/class_excess.parquet")
