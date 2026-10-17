"""Is a house-number "jump" ever copy noise? Train truth (US/India): records with the same name key as an S1 and the same
street key (address words), both with a house number; house relation = same / typo (edit distance 1 on the digit
string) / shift (|d| <= 2) / offset (+1,2,3,4,5,7,9,11,13,21) / jump (anything else). True rate per relation, and the
share of TRUE copies by relation. Then France test accepted pairs (fr14) by the same relation.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/house_jump_truth.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def nkey(s):
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", asc(s).replace(".", ""))) - LEG))


def addr(s):
    a = asc(s)
    m = re.search(r"\b(\d{1,5})\b", a)
    return (m.group(1) if m else None), " ".join(sorted(set(re.findall(r"[a-z]{2,}", a))))


def hrel(h1, h2):
    if h1 == h2:
        return "same"
    d = int(h2) - int(h1)
    if d in OFFS:
        return "offset"
    if abs(d) <= 2:
        return "shift"
    if Levenshtein.distance(h1, h2) == 1:
        return "typo"
    return "jump"


def pairs(s1df, odf):
    s1i = {}
    for e, n, a in s1df.iter_rows():
        h, st = addr(a)
        if h and st:
            s1i.setdefault((nkey(n), st), []).append((e, h))
    out = []
    for e, n, a in odf.iter_rows():
        if not a or not a.strip():
            continue
        h, st = addr(a)
        if not h or not st:
            continue
        for s, hs in s1i.get((nkey(n), st), []):
            out.append((e, s, hrel(hs, h)))
    return pl.DataFrame(out, schema=["m", "s1", "hrel"], orient="row")


gt = pl.read_parquet("local_data/train_pairs.parquet")
for c in ("US", "India"):
    s1 = pl.read_parquet("local_data/train_source1.parquet").filter(pl.col("country") == c).select("entity_id", "business_name", "business_address")
    oth = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (2, 3)], how="vertical_relaxed").filter(pl.col("country") == c) \
            .select("entity_id", "business_name", "business_address")
    x = pairs(s1, oth).join(gt.with_columns(pl.lit(True).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False))
    t = x.group_by("hrel").agg(pl.len().alias("n"), pl.col("y").mean().round(4).alias("true_rate"), pl.col("y").sum().alias("n_true"))
    t = t.with_columns((pl.col("n_true") / pl.col("n_true").sum()).round(4).alias("share_of_copies")).sort("hrel")
    print(f"=== train {c}: same name + same street, by house relation"); print(t)
t1 = pl.read_parquet("local_data/test_source1.parquet").filter(pl.col("country") == "France").select("entity_id", "business_name", "business_address")
t2 = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (2, 3)], how="vertical_relaxed").filter(pl.col("country") == "France") \
       .select("entity_id", "business_name", "business_address")
x = pairs(t1, t2)
g = pl.read_csv("output/v28q_fr14/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("m"))
x = x.join(acc.with_columns(pl.lit(True).alias("acc")), on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
print("=== test France: same name + same street, by house relation (fr14 acceptance)")
print(x.group_by("hrel").agg(pl.len().alias("n"), pl.col("acc").sum().alias("accepted"), pl.col("acc").mean().round(3).alias("acc_rate")).sort("hrel"))
x.write_parquet("local_data/work/fr_house_jump.parquet")
