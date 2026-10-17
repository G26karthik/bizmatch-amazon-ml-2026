"""One-word name swaps vs S1 (name key differs by exactly one word each way), by address relation (same address / house +
decoy offset / other same street), with train truth rates (US/India) and test counts + acceptance (all countries).
Question: are France type-word swaps at the same address ('X Federation' -> 'X Bar') copies or sibling decoys?
Word decoys (holding, ...) sit at house+offset, never at the same address; if sibling decoys follow the decoy path they
would show up at house+offset too.

python scripts/swap_truth.py /work/out_fr7
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    toks = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    street = [" ".join(sorted(t - {h})) if h else "" for t, h in zip(toks, hs)]
    return rec.with_columns(pl.Series("street", street), pl.Series("hn", [int(h) if h and h.isdigit() else None for h in hs], dtype=pl.Int64),
                            pl.Series("nt", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL) for n in rec["business_name"].to_list()]))


def swaps(rec):
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("street") != "")).select(pl.col("rid").alias("s1"), "country", "street", pl.col("hn").alias("hs"),
                                                                              pl.col("nt").alias("nts"), pl.col("entity_id").alias("eid_s1"))
    q = rec.filter((pl.col("src") != 1) & (pl.col("street") != "")).select(pl.col("rid").alias("q"), "country", "street", pl.col("hn").alias("hq"),
                                                                             pl.col("nt").alias("ntq"), pl.col("entity_id").alias("eid_q"))
    d = q.join(s1, on=["country", "street"])
    d = d.with_columns((pl.col("hq") - pl.col("hs")).alias("dh"))
    d = d.filter((pl.col("dh") == 0) | pl.col("dh").is_in(list(OFFS)))
    ok, win, wout = [], [], []
    for a, b in zip(d["ntq"].to_list(), d["nts"].to_list()):
        A, B = set(a), set(b)
        o = len(A - B) == 1 and len(B - A) == 1 and bool(A & B)
        ok.append(o); win.append(next(iter(A - B)) if o else None); wout.append(next(iter(B - A)) if o else None)
    d = d.with_columns(pl.Series("ok", ok), pl.Series("w_in", win), pl.Series("w_out", wout)).filter(pl.col("ok"))
    return d.with_columns(pl.when(pl.col("dh") == 0).then(pl.lit("same")).otherwise(pl.lit("house+off")).alias("arel"))


tr = swaps(prep("/work/base/train/records.parquet"))
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
tr = tr.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
print("TRAIN one-swap records by address relation: true rate")
print(tr.group_by("country", "arel").agg(pl.len(), pl.col("true").mean().round(3).alias("true_rate")).sort("country", "arel"))
for c in ("US", "India"):
    x = tr.filter((pl.col("country") == c) & (pl.col("arel") == "same"))
    print(f"{c} same-address swaps, top swapped-in words among FALSE:", x.filter(~pl.col("true")).group_by("w_in").len().sort("len", descending=True).head(15).rows())
    print(f"{c} same-address swaps, top swapped-in words among TRUE:", x.filter(pl.col("true")).group_by("w_in").len().sort("len", descending=True).head(15).rows())

te = swaps(pl.read_parquet("/work/runfr/test/records.parquet").pipe(lambda _: prep("/work/runfr/test/records.parquet")))
rec_ids = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(rec_ids.rename({"entity_id": "source1_entity_id", "rid": "s1a"}), on="source1_entity_id") \
       .join(rec_ids.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1a")
te = te.join(acc, on="q", how="left").with_columns((pl.col("s1a") == pl.col("s1")).fill_null(False).alias("acc"))
print("\nTEST one-swap records by address relation: count and acceptance to that S1")
print(te.group_by("country", "arel").agg(pl.len(), pl.col("acc").mean().round(3).alias("acc_rate")).sort("country", "arel"))
fr = te.filter(pl.col("country") == "France")
print("France house+offset swaps: top swapped-in words", fr.filter(pl.col("arel") == "house+off").group_by("w_in").len().sort("len", descending=True).head(25).rows())
print("France same-address swaps: top swapped-in words", fr.filter(pl.col("arel") == "same").group_by("w_in").len().sort("len", descending=True).head(25).rows())
us = te.filter(pl.col("country") == "US")
print("US house+offset swaps: top swapped-in words", us.filter(pl.col("arel") == "house+off").group_by("w_in").len().sort("len", descending=True).head(15).rows())
