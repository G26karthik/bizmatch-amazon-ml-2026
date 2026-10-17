"""Accepted pairs whose name key equals the S1's but whose house number differs, by country: house relation classes
(small |d|, decoy offset, digit typo = same digits reordered / one digit dropped or added, prefix, large jump) and the
street relation. US/India train truth gives the true rate of each class. France copies barely change house, so France
accepted mismatches beyond typo classes would be same-name businesses on the same street (false).

python scripts/house_mismatch_accepted.py /work/out_fr10
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


def hclass(hq, hs):
    if not hq or not hs:
        return "house_missing"
    a, b = re.sub(r"\D", "", hq).lstrip("0"), re.sub(r"\D", "", hs).lstrip("0")
    if not a or not b:
        return "house_nondigit"
    if a == b:
        return "same_digits"
    d = int(a) - int(b)
    if d in OFFS:
        return "offset"
    if abs(d) <= 2:
        return "pm1_2"
    if sorted(a) == sorted(b):
        return "transposed"
    if (len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a)))) or (len(b) == len(a) + 1 and any(b[:i] + b[i + 1:] == a for i in range(len(b)))):
        return "digit_drop_add"
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return "digit_sub"
    return "other_small" if abs(d) <= 25 else "large"


def prep(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    nt = [set(re.findall(r"[a-z0-9]+", norm(n).replace(".", ""))) for n in rec["business_name"].to_list()]
    at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    return rec.with_columns(pl.Series("nk", [" ".join(sorted(t - LF - STOP)) for t in nt]),
                            pl.Series("sw", [" ".join(sorted({x for x in t if not x.isdigit()})) for t in at]))


def classify(p, rec):
    x = p.join(rec.select(pl.col("rid").alias("q"), pl.col("nk").alias("nkq"), pl.col("sw").alias("swq"), pl.col("house").alias("hq")), on="q") \
         .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("nk").alias("nks"), pl.col("sw").alias("sws"), pl.col("house").alias("hs")), on="s1")
    x = x.filter((pl.col("nkq") == pl.col("nks")) & (pl.col("nks") != "") & (pl.col("swq") != "") & (pl.col("sws") != ""))
    x = x.with_columns(pl.Series("hcls", [hclass(a, b) for a, b in zip(x["hq"].to_list(), x["hs"].to_list())]),
                       pl.when(pl.col("swq") == pl.col("sws")).then(pl.lit("street_same")).otherwise(pl.lit("street_diff")).alias("street"))
    return x.filter(pl.col("hcls") != "same_digits")


tr = prep("/work/run28/train/records.parquet")
vb = pl.read_parquet("/work/run28/val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1", "prob")
e = tr.select("rid", "entity_id")
gt = pl.read_parquet("/work/data/train_pairs.parquet").join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1") \
       .join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"), pl.lit(True).alias("true"))
x = classify(vb, tr).join(gt, on=["q", "s1"], how="left").with_columns(pl.col("true").fill_null(False))
print("VAL best-S1 candidates, same name key, house differs: true rate and model prob")
print(x.group_by("country", "street", "hcls").agg(pl.len(), pl.col("true").mean().round(3).alias("true_rate"), pl.col("prob").mean().round(3).alias("prob")).sort("country", "street", "hcls"))
te = prep("/work/runfr/test/records.parquet")
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
y = classify(acc, te)
n1 = te.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
print("\nTEST accepted, same name key, house differs: per 1k S1")
print(y.group_by("country", "street", "hcls").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("n_s1") * 1000).round(2).alias("per1k"))
      .drop("n_s1").sort("country", "street", "hcls"))
y.filter(pl.col("country") == "France").select("q", "s1", "street", "hcls").write_parquet("/work/dive/fr_house_mismatch.parquet")
nm = dict(zip(te["rid"].to_list(), te["business_name"].to_list()))
raw = dict(zip(te["rid"].to_list(), te["addr_n"].to_list()))
for q, s, st, hc in y.filter((pl.col("country") == "France") & pl.col("hcls").is_in(["large", "other_small"])).sample(n=12, seed=5).select("q", "s1", "street", "hcls").iter_rows():
    print(f"  [{st} {hc}] {nm[q]} | {raw[q]}  ->  {nm[s]} | {raw[s]}")
