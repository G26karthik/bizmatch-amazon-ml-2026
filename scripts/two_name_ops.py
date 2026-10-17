"""Two name operations. Copies carry at most one name operation (word edit OR legal-form change OR case change); a record
whose name differs from its S1 by a word edit AND a legal-form change is a decoy / other entity. Measures, among pairs
at the S1's street (same house or same address), the true rate of 'word edit + legal change' in US/India train truth and
counts such pairs among test accepted pairs. Word edit = name key (legal/stop removed) differs.

python scripts/two_name_ops.py /work/out_fr8
"""
import re
import sys
import unicodedata

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
out = sys.argv[1]
LF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company",
      "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "the", "of", "and", "com", "www"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def feats(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    toks = [set(re.findall(r"[a-z0-9]+", norm(n).replace(".", ""))) for n in rec["business_name"].to_list()]
    return rec.with_columns(pl.Series("nk", [" ".join(sorted(t - LF - STOP)) for t in toks]),
                            pl.Series("lf", [" ".join(sorted(t & LF)) for t in toks]),
                            pl.Series("ak", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a else "" for a in rec["addr_n"].to_list()]))


def classify(p, rec):
    x = p.join(rec.select(pl.col("rid").alias("q"), pl.col("nk").alias("nkq"), pl.col("lf").alias("lfq"), pl.col("ak").alias("akq"), pl.col("house").alias("hq")), on="q") \
         .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("nk").alias("nks"), pl.col("lf").alias("lfs"), pl.col("ak").alias("aks"), pl.col("house").alias("hs")), on="s1")
    x = x.filter((pl.col("akq") != "") & (pl.col("hq") == pl.col("hs")) & (pl.col("nkq") != pl.col("nks")) & (pl.col("nkq") != "") & (pl.col("nks") != ""))
    sim = [fuzz.token_set_ratio(a, b) for a, b in zip(x["nkq"].to_list(), x["nks"].to_list())]
    return x.with_columns(pl.Series("name_sim", sim),
                          pl.when(pl.col("lfq") == pl.col("lfs")).then(pl.lit("legal_same")).otherwise(pl.lit("legal_changed")).alias("legal"),
                          pl.when(pl.col("akq") == pl.col("aks")).then(pl.lit("addr_exact")).otherwise(pl.lit("addr_edit")).alias("addr"))


# train: all same-house pairs on the same street are too many; use the model's validation candidates (val_best) instead
tr = feats("/work/run28/train/records.parquet")
vb = pl.read_parquet("/work/run28/val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1", "prob")
e = tr.select("rid", "entity_id")
gt = pl.read_parquet("/work/data/train_pairs.parquet").join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1") \
       .join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"), pl.lit(True).alias("true"))
x = classify(vb, tr).join(gt, on=["q", "s1"], how="left").with_columns(pl.col("true").fill_null(False))
print("VAL candidates (best S1), same house, word edit: true rate by legal x addr, and model prob")
print(x.group_by("country", "legal", "addr").agg(pl.len(), pl.col("true").mean().round(4).alias("true_rate"), pl.col("prob").mean().round(3).alias("mean_prob"),
                                                (pl.col("prob") > 0.5).mean().round(3).alias("p>0.5")).sort("country", "legal", "addr"))
te = feats("/work/runfr/test/records.parquet")
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
y = classify(acc, te)
n1 = te.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
print("\nTEST accepted, same house, word edit: counts per 1k S1 by legal x addr")
print(y.group_by("country", "legal", "addr").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("n_s1") * 1000).round(2).alias("per1k")).sort("country", "legal", "addr"))
y.filter((pl.col("country") == "France") & (pl.col("legal") == "legal_changed")).select("q", "s1", "addr", "name_sim").write_parquet("/work/dive/fr_two_name_ops.parquet")
nm = dict(zip(te["rid"].to_list(), te["business_name"].to_list()))
for q, s, ad, ns in y.filter((pl.col("country") == "France") & (pl.col("legal") == "legal_changed")).sample(n=14, seed=3).select("q", "s1", "addr", "name_sim").iter_rows():
    print(f"  [{ad} sim={ns}] {nm[q]}  ->  {nm[s]}")
