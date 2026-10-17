"""Legal-form changes: copy or sibling decoy? Records whose name key (legal/stop removed) equals the S1's, on the S1's
street, by house relation (same house d=0 vs house + decoy offset) and legal relation to the S1 (same, swapped to another
legal form, dropped, added), with acceptance. Decoys only sit at offsets unless a sibling process puts some at d=0;
the offset rows show the decoys' legal mix, d=0 legal-same rows are certain copies. US/India train truth calibrates.

python scripts/legal_swap_france.py /work/out_fr8b
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
out = sys.argv[1]
LF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company",
      "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "the", "of", "and"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    nt = [set(re.findall(r"[a-z0-9]+", norm(n).replace(".", ""))) for n in rec["business_name"].to_list()]
    at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    return rec.with_columns(pl.Series("nk", [" ".join(sorted(t - LF - STOP)) for t in nt]), pl.Series("lf", [" ".join(sorted(t & LF)) for t in nt]),
                            pl.Series("street", [" ".join(sorted(t - {h})) if h else "" for t, h in zip(at, hs)]),
                            pl.Series("hn", [int(h) if h and h.isdigit() else None for h in hs], dtype=pl.Int64))


def pairs(rec):
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("street") != "") & pl.col("hn").is_not_null() & (pl.col("nk") != ""))
    s1 = s1.select(pl.col("rid").alias("s1"), "country", "nk", "street", pl.col("hn").alias("hs"), pl.col("lf").alias("lfs"), pl.col("entity_id").alias("eid_s1"))
    q = rec.filter((pl.col("src") != 1) & (pl.col("street") != "") & pl.col("hn").is_not_null()) \
           .select(pl.col("rid").alias("q"), "country", "nk", "street", pl.col("hn").alias("hq"), pl.col("lf").alias("lfq"), pl.col("entity_id").alias("eid_q"))
    d = q.join(s1, on=["country", "nk", "street"]).with_columns((pl.col("hq") - pl.col("hs")).alias("dh"))
    d = d.filter((pl.col("dh") == 0) | pl.col("dh").is_in(list(OFFS)))
    return d.with_columns(pl.when(pl.col("dh") == 0).then(pl.lit("d0")).otherwise(pl.lit("offset")).alias("arel"),
                          pl.when(pl.col("lfq") == pl.col("lfs")).then(pl.lit("same"))
                          .when(pl.col("lfq") == "").then(pl.lit("dropped"))
                          .when(pl.col("lfs") == "").then(pl.lit("added")).otherwise(pl.lit("swapped")).alias("legal"))


tr = prep("/work/base/train/records.parquet")
d = pairs(tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
print("TRAIN truth: name-key-equal records on the S1 street")
print(d.group_by("country", "arel", "legal").agg(pl.len(), pl.col("true").mean().round(4).alias("true_rate")).sort("country", "arel", "legal"))
te = prep("/work/runfr/test/records.parquet")
e = pairs(te)
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1a"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1a")
e = e.join(acc, on="q", how="left").with_columns((pl.col("s1a") == pl.col("s1")).fill_null(False).alias("acc"))
n1 = te.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
print("\nTEST: counts per 1k S1 and acceptance")
print(e.group_by("country", "arel", "legal").agg(pl.len(), pl.col("acc").mean().round(3).alias("acc_rate")).join(n1, on="country")
      .with_columns((pl.col("len") / pl.col("n_s1") * 1000).round(1).alias("per1k")).drop("n_s1").sort("country", "arel", "legal"))
fr = e.filter((pl.col("country") == "France") & (pl.col("legal") == "swapped"))
print("\nFrance swapped legal pairs (lfq <- lfs) at d0:", fr.filter(pl.col("arel") == "d0").group_by("lfs", "lfq").len().sort("len", descending=True).head(12).rows())
print("France swapped legal pairs at offset:", fr.filter(pl.col("arel") == "offset").group_by("lfs", "lfq").len().sort("len", descending=True).head(12).rows())
e.filter(pl.col("country") == "France").select("q", "s1", "arel", "legal", "lfq", "lfs", "acc").write_parquet("/work/dive/fr_legal_pairs.parquet")
