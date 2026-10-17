"""Deep dive 1: empty-address records whose true S1 has namesakes (other S1s with the same normalised name).

For each such validation record (half A for discovery, half B for confirmation) and each namesake candidate S1, compute
cues and measure how often each cue ranks the true S1 first: chance, the current model (v27 blend), and every cue.
Cues on the record itself: raw-name equality (case-sensitive / insensitive / punctuation-free), legal-suffix form,
casing pattern, brackets. Cues from the S1's *other* records (its likely copies: other records whose best candidate is
that S1 with blended probability >= 0.9): raw similarity to them, shared suffix form / casing / source.

python scripts/dive_namesakes.py --half A
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from unidecode import unidecode

sys.path.insert(0, "src")
from ber.pipeline import SEED, labels  # noqa: E402

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
ap = argparse.ArgumentParser()
ap.add_argument("--half", default="A")
ap.add_argument("--work", type=Path, default=Path("/work/run24"))
ap.add_argument("--tag", default="run24_both")
ap.add_argument("--copies", default="pA", choices=["pA", "blend"], help="how other copies of an S1 are identified")
a = ap.parse_args()

SUF = re.compile(r"(private limited|pvt\.? ?ltd\.?|pvt|limited|ltd\.?|l\.?l\.?c\.?|inc\.?|incorporated|corp\.?|corporation"
                 r"|co\.|company|llp|p\.?c\.?|pllc)\s*[\]\)]?\s*$", re.I)


def suffix(s):
    m = SUF.search(unidecode(s or "").strip())
    return re.sub(r"[^a-z]", "", m.group(1).lower()) if m else ""


def case(s):
    L = [c for c in unidecode(s or "") if c.isalpha()]
    return "n" if not L else "U" if all(c.isupper() for c in L) else "l" if all(c.islower() for c in L) else "M"


def bare(s):
    return re.sub(r"[^a-z0-9]", "", unidecode(s or "").lower())


lab = labels(Path("/work/data"), Path("/work/base")).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
rec = pl.read_parquet("/work/base/train/records.parquet", columns=["business_name", "name_n", "addr_empty", "src"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
half = lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (a.half == "A"))) \
          .select(pl.col("rid").cast(pl.UInt32).alias("s1"))
par = lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32))
sv = pl.read_parquet(a.work / "scores_val.parquet", columns=["q", "s1", "p3"]).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
vb = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
# blended probability for every candidate pair is not saved; use p3 for non-best pairs and the blend for the best pair
d = sv.join(vb.select("q", "s1", pl.col("prob").alias("pb")), on=["q", "s1"], how="left").with_columns(pl.coalesce("pb", "p3").alias("prob"))
d = d.join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("qraw"), pl.col("name_n").alias("qn"), "addr_empty", pl.col("src").alias("qsrc")), on="q") \
     .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("sraw"), pl.col("name_n").alias("sn")), on="s1") \
     .join(par, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
d = d.filter(pl.col("addr_empty") & (pl.col("qn") == pl.col("sn")))
g = d.group_by("q").agg(pl.len().alias("k"), pl.col("y").sum().alias("ny"), pl.col("s1").filter(pl.col("y")).first().alias("true_s1"))
qs = g.filter((pl.col("k") >= 2) & (pl.col("ny") == 1)).join(half.rename({"s1": "true_s1"}), on="true_s1", how="semi")
d = d.join(qs.select("q", "k"), on="q")
print(f"half {a.half}: empty-address records with >=2 namesake candidates incl. the true S1: {qs.height}; mean namesakes {qs['k'].mean():.2f}; chance top-1 {(1 / qs['k']).mean():.3f}")

# the S1s' other likely copies. --copies pA (unbiased): every training record's best stage-A candidate with pA >= 0.9,
# so namesakes that are training S1s get their copies counted too; --copies blend: only validation records (biased).
if a.copies == "pA":
    allp = pl.scan_parquet("/work/run29/train/pruned/*.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "pA").collect()
    bestA = allp.sort("pA", descending=True).group_by("q", maintain_order=True).first()
    src_copies = bestA.filter(pl.col("pA") >= 0.9).select("q", "s1")
    print(f"confident stage-A copies over all training records: {src_copies.height}")
    del allp, bestA
else:
    src_copies = vb.filter(pl.col("prob") >= 0.9).select("q", "s1")
cop = src_copies.select(pl.col("q").alias("c"), "s1").join(
    rec.select(pl.col("rid").alias("c"), pl.col("business_name").alias("craw"), pl.col("src").alias("csrc")), on="c")
cop = cop.join(d.select("s1").unique(), on="s1")
pairs = d.select("q", "s1", "qraw", "qsrc").join(cop, on="s1").filter(pl.col("c") != pl.col("q"))
pairs = pairs.with_columns(pl.Series("cr", cpdist(pairs["qraw"].to_list(), pairs["craw"].to_list(), scorer=fuzz.ratio, workers=2).astype(np.float32)),
                           pl.Series("cbare_eq", [bare(x) == bare(y) for x, y in zip(pairs["qraw"].to_list(), pairs["craw"].to_list())]),
                           pl.Series("craw_eq", [x == y for x, y in zip(pairs["qraw"].to_list(), pairs["craw"].to_list())]),
                           pl.Series("csuf_eq", [suffix(x) == suffix(y) for x, y in zip(pairs["qraw"].to_list(), pairs["craw"].to_list())]),
                           pl.Series("ccase_eq", [case(x) == case(y) for x, y in zip(pairs["qraw"].to_list(), pairs["craw"].to_list())]),
                           (pl.col("qsrc") == pl.col("csrc")).alias("csrc_eq"))
cf = pairs.group_by("q", "s1").agg(pl.len().alias("n_copies"), pl.col("cr").max().alias("copy_max_ratio"), pl.col("cr").mean().alias("copy_mean_ratio"),
                                   pl.col("craw_eq").any().alias("copy_raw_eq"), pl.col("cbare_eq").any().alias("copy_bare_eq"),
                                   pl.col("csuf_eq").mean().alias("copy_suf_share"), pl.col("ccase_eq").mean().alias("copy_case_share"),
                                   pl.col("csrc_eq").sum().alias("copy_same_src"))
d = d.join(cf, on=["q", "s1"], how="left").with_columns(pl.col("n_copies").fill_null(0))
d = d.with_columns(pl.Series("raw_ratio", cpdist(d["qraw"].to_list(), d["sraw"].to_list(), scorer=fuzz.ratio, workers=2).astype(np.float32)),
                   pl.Series("raw_eq", [x == y for x, y in zip(d["qraw"].to_list(), d["sraw"].to_list())]),
                   pl.Series("bare_eq", [bare(x) == bare(y) for x, y in zip(d["qraw"].to_list(), d["sraw"].to_list())]),
                   pl.Series("suf_eq", [suffix(x) == suffix(y) for x, y in zip(d["qraw"].to_list(), d["sraw"].to_list())]),
                   pl.Series("case_eq", [case(x) == case(y) for x, y in zip(d["qraw"].to_list(), d["sraw"].to_list())]),
                   pl.Series("q_bracket", [bool(re.search(r"[\[\]\(\)]", x or "")) for x in d["qraw"].to_list()]))


def top1(col, higher=True):
    x = d.with_columns(pl.col(col).cast(pl.Float64).fill_null(-1e9 if higher else 1e9).alias("_v"))
    x = x.with_columns((pl.col("_v") == (pl.col("_v").max() if higher else pl.col("_v").min()).over("q")).alias("_top"))
    x = x.with_columns(pl.col("_top").sum().over("q").alias("_nt"))
    t = x.filter(pl.col("y"))
    uniq = (t["_top"] & (t["_nt"] == 1)).mean()
    expect = (t["_top"].cast(pl.Float64) / t["_nt"]).mean()  # expected top-1 accuracy with random tie-breaking
    return round(float(uniq), 3), round(float(expect), 3), round(float(x.group_by("q").agg(pl.col("_nt").first())["_nt"].mean()), 2)


rows = [("chance", round(float((1 / qs["k"]).mean()), 3), round(float((1 / qs["k"]).mean()), 3), None)]
for col in ["prob", "p3", "raw_ratio", "raw_eq", "bare_eq", "suf_eq", "case_eq", "n_copies", "copy_max_ratio", "copy_mean_ratio",
            "copy_raw_eq", "copy_bare_eq", "copy_suf_share", "copy_case_share", "copy_same_src"]:
    rows.append((col, *top1(col)))
print(pl.DataFrame(rows, schema=["cue", "true_is_unique_top", "top1_acc_random_ties", "mean_tied_at_top"], orient="row"))
t = d.filter(pl.col("y")); f = d.filter(~pl.col("y"))
print("true vs other namesakes (means):")
for col in ["n_copies", "copy_max_ratio", "copy_raw_eq", "copy_suf_share", "copy_case_share", "raw_eq", "suf_eq", "case_eq"]:
    print(f"  {col:16s} true {float(t[col].cast(pl.Float64).mean()):.3f}  others {float(f[col].cast(pl.Float64).mean()):.3f}")
d.write_parquet(f"/work/dive/namesakes_{a.half}_{a.copies}.parquet")
