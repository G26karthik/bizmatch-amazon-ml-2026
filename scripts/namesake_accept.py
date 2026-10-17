"""Namesake exposure of accepted pairs per country: for each accepted (record, S1) pair, the number of S1s in the country with
the same name key as the chosen S1 (k), split by record address state (empty / same address / other). Pairs whose record
has no address and whose S1 has k namesakes are right only if the model picked the right one of k. Also per-S1
prediction counts vs name-key crowding (France names are low-entropy: '{city} {word}').

python scripts/namesake_accept.py /work/out_fr7
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and",
         "d", "l", "s", "a"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def nkey(s):
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(s))) - LEGAL))


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "addr_empty"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.Series("nk", [nkey(n) for n in rec["business_name"].to_list()]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
s1 = rec.filter(pl.col("src") == 1)
k = s1.group_by("country", "nk").agg(pl.len().alias("k"))
s1k = s1.join(k, on=["country", "nk"]).select(pl.col("rid").alias("s1"), "country", "k", pl.col("addr_n").alias("as"), pl.col("nk").alias("nks"))
d = acc.join(s1k, on="s1").join(rec.select(pl.col("rid").alias("q"), pl.col("addr_n").alias("aq"), pl.col("addr_empty").alias("eq"), pl.col("nk").alias("nkq")), on="q")
d = d.with_columns(pl.when(pl.col("eq").fill_null(False) | (pl.col("aq").fill_null("") == "")).then(pl.lit("empty"))
                   .when(pl.col("aq") == pl.col("as")).then(pl.lit("same_addr")).otherwise(pl.lit("other_addr")).alias("astate"),
                   pl.when(pl.col("k") == 1).then(pl.lit("1")).when(pl.col("k") <= 3).then(pl.lit("2-3")).when(pl.col("k") <= 10).then(pl.lit("4-10"))
                   .otherwise(pl.lit(">10")).alias("kb"))
n_s1 = s1.group_by("country").len().rename({"len": "n_s1"})
t = d.group_by("country", "astate", "kb").len().join(n_s1, on="country").with_columns((pl.col("len") / pl.col("n_s1") * 1000).round(2).alias("per1k"))
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
print(t.pivot(on="country", index=["astate", "kb"], values="per1k").sort("astate", "kb"))
print("share of S1s with a namesake (k>1):", s1k.group_by("country").agg((pl.col("k") > 1).mean().round(4)).rows())
