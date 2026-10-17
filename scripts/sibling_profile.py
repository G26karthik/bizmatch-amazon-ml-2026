"""Profiles of same-address one-word swaps. US train truth gives three references: same-address copies (true), same-address
sibling decoys (false, d=0) and offset decoys (false). France test: same-address type-word swaps (unknown mix), same-
address generic swaps (copies), offset type swaps (decoys). Features: legal relation, lowercase name, cluster (2+ records
sharing S1 + swapped word + house), certain-copy count of the S1. Least-squares decoy share for France same|TYPE using
the US sibling profile as the decoy reference.

python scripts/sibling_profile.py
"""
import re
import sys
import unicodedata

import numpy as np
import polars as pl

sys.argv = sys.argv[:1] + ["/work/out_fr8b"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
LF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company",
      "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}
GEN_FR = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
GEN_US = ["center", "services", "service", "partners", "centre"]


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def legal(s):
    return frozenset(set(re.findall(r"[a-z]+", norm(s).replace(".", ""))) & LF)


def featurize(d, path):
    rec = pl.read_parquet(path, columns=["business_name", "house"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    nm = rec["business_name"].to_list(); hs = rec["house"].to_list()
    rel, low = [], []
    for q, s in zip(d["q"].to_list(), d["s1"].to_list()):
        a, b = legal(nm[q]), legal(nm[s])
        rel.append("same" if a == b else ("rec_none" if not a else ("s1_none" if not b else "diff")))
        low.append(nm[q] == nm[q].lower())
    d = d.with_columns(pl.Series("legal_rel", rel), pl.Series("lower", low), pl.Series("hq", [hs[q] for q in d["q"].to_list()]))
    g = d.group_by("s1", "w_in", "hq").agg(pl.len().alias("gs"))
    return d.join(g, on=["s1", "w_in", "hq"]).with_columns((pl.col("gs") >= 2).alias("clustered"))


def prof(x):
    return x.group_by("cat").agg(pl.len().alias("n"), (pl.col("legal_rel") == "same").mean().alias("legal_same"),
                                 (pl.col("legal_rel") == "rec_none").mean().alias("rec_none"), (pl.col("legal_rel") == "s1_none").mean().alias("s1_none"),
                                 (pl.col("legal_rel") == "diff").mean().alias("diff"), pl.col("lower").mean().alias("lower"),
                                 pl.col("clustered").mean().alias("clustered")).sort("cat")


tr = ns["prep"]("/work/base/train/records.parquet").filter(pl.col("country") == "US")
d = ns["swaps"](tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
d = d.with_columns(pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_US)).then(pl.lit("GEN")).otherwise(pl.lit("OTHER")),
                                  pl.when(pl.col("true")).then(pl.lit("TRUE")).otherwise(pl.lit("FALSE"))], separator="|").alias("cat"))
us = prof(featurize(d, "/work/base/train/records.parquet"))
print("US train truth profiles"); print(us)
f = pl.read_parquet("/work/dive/fr_swaps.parquet").with_columns(
    pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_FR)).then(pl.lit("GEN")).otherwise(pl.lit("TYPE"))], separator="|").alias("cat"))
fr = prof(featurize(f, "/work/runfr/test/records.parquet"))
print("France test profiles"); print(fr)
feats = ["legal_same", "rec_none", "s1_none", "diff", "lower"]
cop = fr.filter(pl.col("cat") == "same|GEN").select(feats).to_numpy()[0]
sib_us = us.filter(pl.col("cat") == "same|OTHER|FALSE").select(feats).to_numpy()[0]
cop_us = us.filter(pl.col("cat") == "same|OTHER|TRUE").select(feats).to_numpy()[0]
mix = fr.filter(pl.col("cat") == "same|TYPE").select(feats).to_numpy()[0]
for name, dec in (("US sibling profile", sib_us), ("France offset decoys", fr.filter(pl.col("cat") == "offset|TYPE").select(feats).to_numpy()[0])):
    # mix = pi*dec + (1-pi)*cop -> least squares pi
    a = dec - cop; b = mix - cop
    pi = float(a @ b / (a @ a))
    print(f"decoy share of France same|TYPE with {name} as decoy reference: {pi:.3f}; per-feature:",
          {k: round(float((m - c) / (dd - c)), 3) if abs(dd - c) > 1e-3 else None for k, m, c, dd in zip(feats, mix, cop, dec)})
