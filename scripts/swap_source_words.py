"""For France SWAP pairs to a watched word (from word_add_test.py output), which S1 word was replaced, with acceptance,
bracket share and cap test per (replaced word -> new word); plus examples.

python scripts/swap_source_words.py compagnie frs service
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
words = sys.argv[1:]


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


x = pl.read_parquet("/work/lead/fr_word_add.parquet").filter((pl.col("rel") == "SWAP") & pl.col("word").is_in(words))
te = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name", "addr_n"]).with_row_index("rid")
nm = te["business_name"].to_list(); ad = te["addr_n"].to_list()
rep = []
for q, s in zip(x["q"].to_list(), x["s1"].to_list()):
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
    rep.append(next(iter(B - A)) if len(B - A) == 1 else "?")
x = x.with_columns(pl.Series("old", rep))
t = x.group_by("word", "old", "hrel").agg(pl.len().alias("n"), pl.col("acc").sum().alias("acc"), pl.col("bracket").mean().round(3).alias("brk"),
                                           pl.col("n23").sum().alias("n23"), pl.col("at_cap").sum().alias("ncap")) \
     .filter(pl.col("n") >= 20).sort("n", descending=True)
print(t.head(60))
for w in words:
    print(f"\n--- examples {w} same house accepted")
    for q, s in x.filter((pl.col("word") == w) & (pl.col("hrel") == "same") & pl.col("acc")).select("q", "s1").head(12).iter_rows():
        print(f"   {nm[s]!r:45} | {nm[q]!r:45} | {ad[s]} | {ad[q]}")
