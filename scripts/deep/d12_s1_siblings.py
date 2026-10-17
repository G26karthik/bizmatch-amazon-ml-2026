"""Deep dive 12: S1s sharing an address (normalized) with another S1, by country and split; name relation of such
pairs (one-word swap / same key / other) and how fr16 splits records between them.

uv run --no-project --with polars==1.44.2 python scripts/deep/d12_s1_siblings.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def akey(s):
    return " ".join(sorted(toks(s)))


for split in ("train", "test"):
    s1 = pl.read_parquet(f"local_data/{split}_source1.parquet").with_columns(pl.col("business_address").map_elements(akey, return_dtype=pl.String).alias("ak"))
    grp = s1.group_by("country", "ak").agg(pl.len().alias("n"), pl.col("business_name").alias("names"), pl.col("entity_id").alias("ids"))
    print(split, grp.filter(pl.col("n") >= 2).group_by("country").agg(pl.col("n").sum().alias("S1s_in_shared_addr"), pl.len().alias("groups")).join(
        s1.group_by("country").len(), on="country").with_columns((pl.col("S1s_in_shared_addr") / pl.col("len")).round(4).alias("share")).sort("country"))
    rel = Counter()
    for c, names in grp.filter(pl.col("n") == 2).select("country", "names").iter_rows():
        a, b = set(toks(names[0])), set(toks(names[1]))
        r = "same_key" if a == b else ("one_word_swap" if len(a - b) == 1 and len(b - a) == 1 else ("subset" if a < b or b < a else "other"))
        rel[(c, r)] += 1
    print(split, sorted(rel.items()))
    ex = grp.filter((pl.col("n") == 2) & (pl.col("country") == "France")).head(8) if split == "test" else grp.filter(pl.col("n") == 2).head(4)
    for r in ex.iter_rows(named=True):
        print("   ", r["country"], r["names"], "|", r["ak"][:60])
