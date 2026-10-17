"""Copy-type mix: classify each (record, S1) pair by the kind of name change and address change, then compare
  - train TRUTH pairs per country (the generator's copy mix),
  - test ACCEPTED pairs per country (our predictions).
US test F0.5 equals validation (LB diagnostic), so US accepted should reproduce US truth; France accepted vs the
truth mix shows which copy types France misses (deficit) or over-accepts (excess). Rates are per S1.

python scripts/op_mix.py /work/out_fr7
"""
import re
import sys
import unicodedata

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "gmbh", "sl", "spa", "the"}
STOPW = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "of", "and", "s", "a", "u", "r", "e"}
DECOY = {"holding", "holdings", "participations", "international", "distribution", "groupe", "group", "developpement", "france", "partners",
         "midtown", "northside", "southside", "greater", "riverside", "downtown", "metro", "eastside", "westside", "west", "east", "north",
         "south", "valley", "harbor", "uptown", "summit", "coastal", "central", "lakeside", "public", "overseas", "ventures", "enterprises",
         "industries", "exports"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def ntoks(s):
    return set(re.findall(r"[a-z0-9]+", norm(s))) - STOPW


def name_op(q, s):
    tq, ts = ntoks(q), ntoks(s)
    cq, cs = tq - LEGAL, ts - LEGAL
    if tq == ts:
        return "identical"
    if cq == cs:
        return "legal_change"
    raw = norm(q).strip()
    if (".com" in raw or "@" in raw or "#" in raw or (" " not in raw and len(raw) > 8)) and fuzz.partial_ratio(raw.replace(" ", ""), "".join(sorted(cs))) >= 0:
        if ".com" in raw or "@" in raw or "#" in raw:
            return "web_handle"
    letters = re.sub(r"[^a-z]", "", norm(q))
    initials = "".join(w[0] for w in re.findall(r"[a-z0-9]+", norm(s)) if w not in LEGAL and w not in STOPW)
    if 1 < len(letters) <= 4 and letters == initials[:len(letters)]:
        return "initials"
    if cq > cs and len(cq - cs) == 1:
        w = next(iter(cq - cs))
        return "add_decoyword" if w in DECOY else "add_word"
    if cq < cs and len(cs - cq) == 1:
        return "drop_word"
    if len(cq - cs) == 1 and len(cs - cq) == 1:
        a, b = next(iter(cq - cs)), next(iter(cs - cq))
        if fuzz.ratio(a, b) >= 70:
            return "typo"
        return "swap_decoyword" if a in DECOY else "swap_word"
    if not (cq & cs):
        return "unrelated_name" if fuzz.ratio(" ".join(sorted(cq)), " ".join(sorted(cs))) < 60 else "garbled"
    if fuzz.token_set_ratio(q, s) >= 85:
        return "garbled"
    return "multi_change"


def addr_op(aq, hq, as_, hs):
    if not aq or not as_:
        return "missing"
    tq, ts = set(re.findall(r"[a-z0-9]+", norm(aq))), set(re.findall(r"[a-z0-9]+", norm(as_)))
    if tq == ts:
        return "same"
    wq, ws = {t for t in tq if not t.isdigit()}, {t for t in ts if not t.isdigit()}
    same_house = bool(hq) and hq == hs
    if wq == ws:
        return "house_change" if not same_house else "same_words"
    if same_house:
        return "street_edit" if len(wq ^ ws) <= 2 else "street_multi"
    return "house+street_change"


def classify(df):
    n = [name_op(q, s) for q, s in zip(df["nq"].to_list(), df["ns"].to_list())]
    a = [addr_op(*r) for r in zip(df["aq"].to_list(), df["hq"].to_list(), df["as"].to_list(), df["hs"].to_list())]
    return df.with_columns(pl.Series("name_op", n), pl.Series("addr_op", a))


def pairs_frame(rec, pairs):
    cols = {"business_name": "n", "addr_n": "a", "house": "h"}
    qd = rec.select(pl.col("rid").alias("q"), *[pl.col(k).alias(v + "q") for k, v in cols.items()])
    sd = rec.select(pl.col("rid").alias("s1"), "country", *[pl.col(k).alias(v + "s") for k, v in cols.items()]).rename({"as": "as"})
    return pairs.join(qd, on="q").join(sd, on="s1")


# --- train truth (sample) ---
trec = pl.read_parquet("/work/base/train/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
         .with_columns(pl.col("rid").cast(pl.UInt32))
tp = pl.read_parquet("/work/data/train_pairs.parquet")
e = trec.select("rid", "entity_id")
tpairs = tp.join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"))
tpairs = tpairs.sample(n=400000, seed=1)
tr = classify(pairs_frame(trec, tpairs))
ns1_train = trec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
scale_t = tp.height / 400000
# --- test accepted ---
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
e2 = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t")
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .join(e2.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(e2.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
te = classify(pairs_frame(rec, p))
ns1_test = rec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})


def rates(df, ns1, scale, label):
    return df.group_by("country", "name_op", "addr_op").len().join(ns1, on="country") \
             .with_columns((pl.col("len") * scale / pl.col("n_s1") * 1000).round(2).alias(label)).select("country", "name_op", "addr_op", label)


rt = rates(tr, ns1_train, scale_t, "truth_per1k")
ra = rates(te, ns1_test, 1.0, "accepted_per1k")
us_truth = rt.filter(pl.col("country") == "US").drop("country").rename({"truth_per1k": "US_truth"})
in_truth = rt.filter(pl.col("country") == "India").drop("country").rename({"truth_per1k": "IN_truth"})
tab = us_truth.join(in_truth, on=["name_op", "addr_op"], how="full", coalesce=True)
for c in ("US", "India", "France"):
    tab = tab.join(ra.filter(pl.col("country") == c).drop("country").rename({"accepted_per1k": f"{c}_acc"}), on=["name_op", "addr_op"], how="full", coalesce=True)
tab = tab.fill_null(0).with_columns((pl.col("France_acc") - (pl.col("US_truth") + pl.col("IN_truth")) / 2).round(2).alias("FR_minus_truth")) \
         .sort("US_truth", descending=True)
print("pairs per 1000 S1 by (name change, address change): train truth US/India vs test accepted")
print(tab.head(60))
print("totals per 1000 S1:", tab.select([pl.col(c).sum().round(1) for c in ("US_truth", "IN_truth", "US_acc", "India_acc", "France_acc")]).rows())
tab.write_parquet("/work/dive/op_mix.parquet")
