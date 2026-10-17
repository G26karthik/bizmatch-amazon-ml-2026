"""Decoy-cluster test. Copies of one S1 get independent noise, so two records assigned to one S1 rarely share the same
NON-S1 name key; a decoy business (different entity, 1-3 noisy copies, e.g. a sibling name at the S1's address) shows up as
2+ records sharing a name key that differs from the S1's. Train truth gives the legit rate (deterministic noise such as
drop-word or initials); test US (F = validation) shows the calibrated accepted rate; France is compared.

python scripts/shared_noise.py /work/out_fr7
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def nk(s):
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(s))) - LEGAL))


def shared(pairs, rec, label):
    """pairs: q, s1 (rids). Returns per-country stats of shared non-S1 name keys among records of one S1."""
    x = pairs.join(rec.select(pl.col("rid").alias("q"), pl.col("nk").alias("nkq")), on="q") \
             .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("nk").alias("nks")), on="s1")
    x = x.filter((pl.col("nkq") != pl.col("nks")) & (pl.col("nkq") != ""))
    grp = x.group_by("country", "s1", "nkq").agg(pl.len().alias("n"), pl.col("q"))
    rep = grp.filter(pl.col("n") >= 2)
    n_s1 = rec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
    st = rep.group_by("country").agg(pl.len().alias("groups"), pl.col("n").sum().alias("records")).join(n_s1, on="country") \
            .with_columns((pl.col("groups") / pl.col("n_s1") * 1000).round(2).alias("groups_per1k_S1"),
                          (pl.col("records") / pl.col("n_s1") * 1000).round(2).alias("records_per1k_S1"))
    print(f"\n{label}: 2+ records of one S1 sharing a name key that differs from the S1's")
    print(st.sort("country"))
    return rep


tr = pl.read_parquet("/work/base/train/records.parquet", columns=["entity_id", "src", "country", "business_name"]).with_row_index("rid") \
       .with_columns(pl.col("rid").cast(pl.UInt32))
tr = tr.with_columns(pl.Series("nk", [nk(n) for n in tr["business_name"].to_list()]))
e = tr.select("rid", "entity_id")
tp = pl.read_parquet("/work/data/train_pairs.parquet")
tpairs = tp.join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"))
shared(tpairs, tr, "TRAIN truth")

rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.Series("nk", [nk(n) for n in rec["business_name"].to_list()]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .filter(pl.col("matched_entity_ids") != "") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
rep = shared(p, rec, "TEST accepted")
fr = rep.filter(pl.col("country") == "France").sample(n=15, seed=2)
names = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list())); addrs = dict(zip(rec["rid"].to_list(), rec["business_address"].to_list()))
for s, qs in zip(fr["s1"].to_list(), fr["q"].to_list()):
    print(f"  S1: {names[s]} | {addrs[s]}")
    for q in qs:
        print(f"      {names[q]} | {addrs[q]}")
