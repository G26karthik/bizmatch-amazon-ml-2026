"""One-operation test. For record-S1 pairs whose name key (legal/stop words removed) is equal, split by legal-form
relation (identical legal vs legal changed/dropped/added) and report the share with an exactly equal address (token set)
and with an address change. If France copies carry one noise operation, legal-changed copies keep the address exactly.
US train truth (copies vs decoys) and US/France test accepted pairs.

python scripts/one_op_test.py /work/out_fr8
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
out = sys.argv[1]
LF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company",
      "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "the", "of", "and"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def feats(rec):
    toks = [set(re.findall(r"[a-z0-9]+", norm(n).replace(".", ""))) for n in rec["business_name"].to_list()]
    return rec.with_columns(pl.Series("nk", [" ".join(sorted(t - LF - STOP)) for t in toks]),
                            pl.Series("lf", [" ".join(sorted(t & LF)) for t in toks]),
                            pl.Series("ak", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a else "" for a in rec["addr_n"].to_list()]),
                            pl.Series("hn", rec["house"].to_list()))


def classify(p, rec):
    x = p.join(rec.select(pl.col("rid").alias("q"), pl.col("nk").alias("nkq"), pl.col("lf").alias("lfq"), pl.col("ak").alias("akq"), pl.col("hn").alias("hq")), on="q") \
         .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("nk").alias("nks"), pl.col("lf").alias("lfs"), pl.col("ak").alias("aks"), pl.col("hn").alias("hs")), on="s1")
    x = x.filter((pl.col("nkq") == pl.col("nks")) & (pl.col("nks") != "") & (pl.col("akq") != "") & (pl.col("aks") != ""))
    return x.with_columns(pl.when(pl.col("lfq") == pl.col("lfs")).then(pl.lit("legal_same")).otherwise(pl.lit("legal_changed")).alias("legal"),
                          pl.when(pl.col("akq") == pl.col("aks")).then(pl.lit("addr_exact"))
                          .when(pl.col("hq") == pl.col("hs")).then(pl.lit("addr_edit_same_house")).otherwise(pl.lit("addr_house_diff")).alias("addr"))


def show(x, label, extra=()):
    t = x.group_by(*extra, "legal", "addr").len().with_columns((pl.col("len") / pl.col("len").sum().over(*extra, "legal")).round(4).alias("share"))
    print(f"\n=== {label}")
    print(t.pivot(on="addr", index=[*extra, "legal"], values="share").fill_null(0).sort(*extra, "legal"))


tr = feats(pl.read_parquet("/work/base/train/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"])
           .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
gt = pl.read_parquet("/work/data/train_pairs.parquet")
e = tr.select("rid", "entity_id")
tp = gt.join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"))
show(classify(tp, tr), "TRAIN truth copies (name key equal)", ["country"])
te = feats(pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"])
           .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
x = classify(acc, te)
show(x, "TEST accepted (name key equal)", ["country"])
print("\ncounts, France accepted:", x.filter(pl.col("country") == "France").group_by("legal", "addr").len().sort("legal", "addr").rows())
