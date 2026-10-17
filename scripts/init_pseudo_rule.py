"""Initials and pseudo-name records at a single-S1 address. Pairs every S2/S3 record with the S1s at the same street +
house (normalised address tokens, house removed from the street key); keeps addresses with exactly one S1.
  INIT  : record letters (2-5) = initials of the S1's core name (ber.normalize.name_core), or of all its tokens, or a prefix
  PSEUDO: record name key (legal/stop removed) shares no token with the S1's, 1-2 tokens, no decoy word, not initials
Validation (US/India, labels): true rate, production acceptance. Test France: counts, unassigned in the submission, prob.
Writes France adds (unassigned records, S1 under its source cap) to /work/lead/fr_init_pseudo_adds.parquet.

python scripts/init_pseudo_rule.py /work/out_fr13
"""
import re
import sys
import unicodedata
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.normalize import name_core  # noqa: E402
from ber.pipeline import labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
out = sys.argv[1]
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l",
       "dba", "aka", "ta", "t", "b", "k"}
DW = {"holding", "participations", "distribution", "international", "groupe", "developpement", "southside", "eastgate", "northside", "midtown", "greater"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def is_acr(qn, sn):
    q = re.sub(r"[^a-z]", "", norm(qn))
    if not (1 < len(q) <= 5) or len(re.findall(r"[a-z0-9]+", norm(qn))) > 2:
        return False
    ini = "".join(t[0] for t in name_core(sn or "").split())
    full = "".join(t[0] for t in re.findall(r"[a-z0-9]+", norm(sn)))
    return q == ini or q == full or (len(q) >= 2 and ini.startswith(q))


def classify(rec, pairs_only_q=None):
    nm = rec["business_name"].to_list(); src = rec["src"].to_list()
    at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    key = [(" ".join(sorted(t - {h})), h) if (h and t) else None for t, h in zip(at, hs)]
    by = {}
    for i, k in enumerate(key):
        if src[i] == 1 and k:
            by.setdefault(k, []).append(i)
    rows = []
    for i, k in enumerate(key):
        if src[i] == 1 or not k or (pairs_only_q is not None and i not in pairs_only_q):
            continue
        s1s = by.get(k, [])
        if len(s1s) != 1:
            continue
        s = s1s[0]
        A = set(re.findall(r"[a-z0-9]+", norm(nm[i]))) - LEG
        B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
        if not A or not B:
            continue
        if is_acr(nm[i], nm[s]):
            c = "INIT"
        elif not (A & B) and len(A) <= 2 and not (A & DW):
            c = "PSEUDO"
        else:
            continue
        rows.append((i, s, c, src[i]))
    return pl.DataFrame(rows, schema=["q", "s1", "cls", "src"], orient="row").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))


# ---- validation
work = Path("/work/run28")
lab = labels(Path("/work/data"), work)
tr = pl.read_parquet(work / "train" / "records.parquet", columns=["business_name", "src", "country", "addr_n", "house"])
valq = set(lab.filter(pl.col("is_val") & (pl.col("src") != 1))["rid"].to_list())
x = classify(tr, valq)
par = lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32))
ch = _chosen(pl.read_parquet(work / "val_best_run28_7q.parquet"), {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
x = x.join(par, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"), pl.col("parent").is_null().alias("orphan"))
x = x.join(ch.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
x = x.join(ch.select("q").with_columns(pl.lit(True).alias("assigned")), on="q", how="left").with_columns(pl.col("assigned").fill_null(False))
x = x.join(tr.select(pl.col("country")).with_row_index("q").with_columns(pl.col("q").cast(pl.UInt32)), on="q")
print("=== validation (single-S1 address)")
print(x.group_by("country", "cls", "assigned").agg(pl.len().alias("n"), pl.col("y").mean().round(4).alias("true_rate"), pl.col("orphan").mean().round(3).alias("orphan"),
      pl.col("acc").mean().round(3).alias("accepted")).sort("country", "cls", "assigned"))
del tr
# ---- test France
te = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name", "src", "country", "addr_n", "house"]).with_row_index("rid") \
       .with_columns(pl.col("rid").cast(pl.UInt32))
fr = te.filter(pl.col("country") == "France")
frid = fr["rid"].to_list()
y = classify(fr.drop("rid", "entity_id"))
y = y.with_columns(pl.Series("q", [frid[i] for i in y["q"].to_list()], dtype=pl.UInt32), pl.Series("s1", [frid[i] for i in y["s1"].to_list()], dtype=pl.UInt32))
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32).alias("best_s1"), "prob")
y = y.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
y = y.join(acc.select("q").with_columns(pl.lit(True).alias("assigned")), on="q", how="left").with_columns(pl.col("assigned").fill_null(False))
y = y.join(tb, on="q", how="left")
print("\n=== test France (single-S1 address)")
print(y.group_by("cls", "acc", "assigned").agg(pl.len().alias("n"), (pl.col("best_s1") == pl.col("s1")).mean().round(3).alias("best_is_s1"),
      pl.col("prob").mean().round(3).alias("mean_p_best")).sort("cls", "acc", "assigned"))
srcd = dict(zip(te["rid"].to_list(), te["src"].to_list()))
cnt = acc.with_columns(pl.Series("src", [srcd[q] for q in acc["q"].to_list()], dtype=pl.Int8)).group_by("s1", "src").len()
adds = y.filter(~pl.col("assigned")).with_columns(pl.col("src").cast(pl.Int8)).join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0))
adds = adds.filter(((pl.col("src") == 2) & (pl.col("len") < 5)) | ((pl.col("src") == 3) & (pl.col("len") < 6))).unique(subset="q", keep="none")
print("France adds by class:", adds.group_by("cls").len().rows())
nm = te["business_name"].to_list()
for c in ("INIT", "PSEUDO"):
    print(f"examples {c}")
    for q, s in adds.filter(pl.col("cls") == c).head(8).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:44} | {nm[q]!r}")
adds.select("q", "s1", "cls", "prob").write_parquet("/work/lead/fr_init_pseudo_adds.parquet")
