"""Deep dive 29: accepted pairs at the strict address of their S1 (all house numbers equal, address token Jaccard >= 0.6)
by name similarity bin, per 1k S1s: test (fr22 accepted) vs train (true copies, all S1 visible). An excess in test in
the near-duplicate bins would mean test's extra decoys (near-duplicate names at the same address) are being accepted.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d29_accepted_neardup.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def same_addr(a, b):
    A, B = set(toks(a)), set(toks(b))
    da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
    return bool(da) and da == db and len(A & B) / max(1, len(A | B)) >= 0.6


def rates(split, pairs):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); ct = dict(zip(rec["entity_id"], rec["country"]))
    n1 = rec.filter(pl.col("entity_id").str.starts_with("S1")).group_by("country").len()
    rows = []
    for s, m in pairs:
        if ct[s] == "France" or not same_addr(ad[m], ad[s]):
            continue
        rows.append((ct[s], fuzz.token_set_ratio(" ".join(toks(nm[m])), " ".join(toks(nm[s])))))
    x = pl.DataFrame(rows, schema=["country", "sim"], orient="row").with_columns(pl.col("sim").cut([50, 70, 80, 90, 99]).alias("simb"))
    return x.group_by("country", "simb").len().join(n1.rename({"len": "S1s"}), on="country").with_columns((pl.col("len") / pl.col("S1s") * 1000).round(2).alias(f"per1k_{split}"))


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = rates("train", zip(tp["s1"].to_list(), tp["m"].to_list()))
g = pl.read_csv("output/v28q_fr22/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
te = rates("test", zip(a["source1_entity_id"].to_list(), a["matched_entity_ids"].to_list()))
print(tr.select("country", "simb", "per1k_train").join(te.select("country", "simb", "per1k_test"), on=["country", "simb"]).with_columns(
    (pl.col("per1k_test") / pl.col("per1k_train")).round(3).alias("ratio")).sort("country", "simb"))
