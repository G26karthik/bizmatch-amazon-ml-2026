"""House-number difference profile. Records with the S1's name key on the S1's street, by signed house difference
d = house(record) - house(S1). Decoys only use d in {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21}; copy house noise need not be
signed, so N(-d) estimates copies at +d and N(+d) - N(-d) the decoys. Train truth (US/India) validates the logic; test
acceptance per country shows whether France copies with house noise are rejected as decoys.

python scripts/house_diff.py /work/out_fr7
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    toks = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    return rec.with_columns(pl.Series("street", [" ".join(sorted(t - {h})) if h else "" for t, h in zip(toks, hs)]),
                            pl.Series("hn", [int(h) if h and h.isdigit() else None for h in hs], dtype=pl.Int64),
                            pl.Series("nk", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL)) for n in rec["business_name"].to_list()]))


def pairs(rec):
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("street") != "") & pl.col("hn").is_not_null())
    s1 = s1.join(s1.group_by("country", "nk", "street").len(), on=["country", "nk", "street"]).filter(pl.col("len") == 1)
    s1 = s1.select(pl.col("rid").alias("s1"), "country", "nk", "street", pl.col("hn").alias("hs"), pl.col("entity_id").alias("eid_s1"))
    q = rec.filter((pl.col("src") != 1) & (pl.col("street") != "") & pl.col("hn").is_not_null()) \
           .select(pl.col("rid").alias("q"), "country", "nk", "street", pl.col("hn").alias("hq"), pl.col("entity_id").alias("eid_q"))
    d = q.join(s1, on=["country", "nk", "street"]).with_columns((pl.col("hq") - pl.col("hs")).alias("d"))
    return d.filter(pl.col("d").abs() <= 25)


tr = pairs(prep("/work/base/train/records.parquet"))
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
tr = tr.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
te = pairs(prep("/work/runfr/test/records.parquet"))
rec_ids = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(rec_ids.rename({"entity_id": "source1_entity_id", "rid": "s1a"}), on="source1_entity_id") \
       .join(rec_ids.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1a")
te = te.join(acc, on="q", how="left").with_columns((pl.col("s1a") == pl.col("s1")).fill_null(False).alias("acc"))
a = tr.group_by("country", "d").agg(pl.len().alias("n"), pl.col("true").sum().alias("true_n")).sort("d")
b = te.group_by("country", "d").agg(pl.len().alias("n"), pl.col("acc").sum().alias("acc_n")).sort("d")
print("d=0:", b.filter(pl.col("d") == 0).rows(), a.filter(pl.col("d") == 0).rows())


def show(label, df, cols):
    print(f"\n{label}")
    for row in df.sort("d").iter_rows(named=True):
        print("  d=%+3d " % row["d"] + "  ".join(f"{c}={row[c]}" for c in cols))


for c in ("US", "India"):
    x = a.filter(pl.col("country") == c).join(b.filter(pl.col("country") == c).drop("country"), on="d", how="full", suffix="_test", coalesce=True)
    show(c, x.filter(pl.col("d").abs() <= 22), ["n", "true_n", "n_test", "acc_n"])
show("France", b.filter((pl.col("country") == "France") & (pl.col("d").abs() <= 22)), ["n", "acc_n"])
