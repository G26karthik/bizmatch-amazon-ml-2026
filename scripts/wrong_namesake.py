"""Wrong-namesake check: for each accepted pair, is there a DIFFERENT S1 with the same name (token set) whose address
matches the record better (same normalised address, or same house + street) than the chosen S1? Count-based checks
cannot see such swaps (predicted counts stay right while precision and recall both drop).

python scripts/wrong_namesake.py /work/out_fr7 [--write /work/dive/swap_fix.parquet]
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e",
        "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "the", "of", "and", "co", "lp", "llp", "corporation", "incorporated", "company"}


def key(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", s)) - STOP))


def street(addr, house):
    a = unicodedata.normalize("NFKD", addr or "").encode("ascii", "ignore").decode().lower()
    return " ".join(t for t in re.findall(r"[a-z]+", a))


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.Series("nk", [key(n) for n in rec["business_name"].to_list()]),
                       pl.Series("st", [street(a, h) for a, h in zip(rec["addr_n"].to_list(), rec["house"].to_list())]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t")
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country", "nk", pl.col("addr_n").alias("a1"), pl.col("house").alias("h1"), pl.col("st").alias("st1"))
q = rec.filter(pl.col("src") != 1).select(pl.col("rid").alias("q"), pl.col("addr_n").alias("aq"), pl.col("house").alias("hq"), pl.col("st").alias("stq"), pl.col("nk").alias("nkq"))
d = p.join(q, on="q").join(s1, on="s1")


def match_level(aq, hq, stq, a1, h1, st1):
    if not aq:
        return 0
    if aq == a1:
        return 3
    if hq and h1 and hq == h1 and stq == st1:
        return 2
    if hq and h1 and hq == h1:
        return 1
    return 0


d = d.with_columns(pl.Series("lvl", [match_level(*r) for r in zip(d["aq"].to_list(), d["hq"].to_list(), d["stq"].to_list(), d["a1"].to_list(), d["h1"].to_list(), d["st1"].to_list())]))
# alternative namesakes: other S1s, same country, same name key as the chosen S1
alt = d.select("q", "s1", "country", "nk", "aq", "hq", "stq", "lvl").join(
    s1.rename({"s1": "s1b", "a1": "a1b", "h1": "h1b", "st1": "st1b"}), on=["country", "nk"]).filter(pl.col("s1b") != pl.col("s1"))
alt = alt.with_columns(pl.Series("lvlb", [match_level(*r) for r in zip(alt["aq"].to_list(), alt["hq"].to_list(), alt["stq"].to_list(),
                                                                          alt["a1b"].to_list(), alt["h1b"].to_list(), alt["st1b"].to_list())]))
better = alt.filter(pl.col("lvlb") > pl.col("lvl")).sort("lvlb", descending=True).unique(subset=["q"], keep="first")
print(f"{out}: accepted pairs {d.height}; pairs whose chosen S1 has same-name namesakes {alt.select('q').n_unique()}")
print("pairs with a namesake S1 matching the record's address better, by country and (chosen level -> better level):")
print(better.group_by("country", "lvl", "lvlb").len().sort("len", descending=True))
ex = better.join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("nq")), on="q").filter(pl.col("country") == "France")
print(ex.sample(n=min(12, ex.height), seed=1).select("nq", "aq", "a1b", "lvl", "lvlb"))
if write:
    better.select("q", "s1", "s1b", "country", "lvl", "lvlb").write_parquet(write)
