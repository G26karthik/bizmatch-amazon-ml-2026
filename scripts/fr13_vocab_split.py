"""Contested French same-house type swaps split by the new word's count in the French S1 vocabulary (0 = non-word, a
typo of the S1 word -> copy) x blended prob x fr7/fr12 status; bracket-share decoy estimate per cell.
d = (0.0418 - brk) / (0.0418 - 0.0090).

python scripts/fr13_vocab_split.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list()
fr = rec.filter(pl.col("country") == "France")
voc1 = Counter(w for n in fr.filter(pl.col("src") == 1)["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", norm(n))))
vocall = Counter(w for n in fr["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", norm(n))))
eid = rec.select("rid", "entity_id")


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


a7, a12 = accepted("/work/out_fr7"), accepted("/work/out_fr12")
bm = pl.read_parquet("/work/lead/fr_bracket_mix.parquet").filter((pl.col("kind") == "type") & (pl.col("hrel") == "same")).select("q", "s1", "bracket")
cb = pl.read_parquet("/work/lead/fr_swap_brk.parquet").filter(pl.col("cat") == "C_edit_type").select("q", "s1", "bracket")
x = pl.concat([bm, cb]).unique(["q", "s1"])
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
x = x.join(tb, on=["q", "s1"], how="left")
x = x.join(a7.with_columns(pl.lit(True).alias("k7")), on=["q", "s1"], how="left").join(a12.with_columns(pl.lit(True).alias("k12")), on=["q", "s1"], how="left") \
     .with_columns(pl.col("k7").fill_null(False), pl.col("k12").fill_null(False))
x = x.with_columns(pl.when(pl.col("k12")).then(pl.lit("kept")).when(pl.col("k7")).then(pl.lit("removed")).otherwise(pl.lit("rejected")).alias("status"))
v1, va, sim, new, old = [], [], [], [], []
for q, s in zip(x["q"].to_list(), x["s1"].to_list()):
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
    wi = next(iter(A - B)) if len(A - B) == 1 else ""
    wo = next(iter(B - A)) if len(B - A) == 1 else ""
    v1.append(voc1.get(wi, 0)); va.append(vocall.get(wi, 0)); new.append(wi); old.append(wo)
    sim.append(fuzz.ratio(wi, wo) if wi and wo else 0)
x = x.with_columns(pl.Series("voc1", v1), pl.Series("vocall", va), pl.Series("sim", sim), pl.Series("new", new), pl.Series("old", old))
x = x.with_columns(pl.col("voc1").cut([0, 3, 30]).alias("vbin"), pl.col("prob").fill_null(-1).cut([0.5, 0.99, 0.9999]).alias("pbin"),
                   pl.col("sim").cut([30, 50, 70]).alias("simbin"))


def show(df, by, title):
    t = df.group_by(by).agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk")) \
          .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).sort(by)
    print(f"\n=== {title}"); print(t)


show(x, ["status", "vbin"], "all contested by status x S1-vocab count of the new word")
show(x.filter(pl.col("status") == "removed"), ["vbin", "pbin"], "removed: vocab x prob")
show(x.filter(pl.col("status") == "removed"), ["vbin", "simbin"], "removed: vocab x similarity(new, old)")
show(x.filter(pl.col("status") == "kept"), ["vbin", "pbin"], "kept: vocab x prob")
for v in ("(-inf, 0]", "(0, 3]"):
    print(f"\nexamples removed vbin {v}")
    for r in x.filter((pl.col("status") == "removed") & (pl.col("vbin").cast(pl.String) == v)).sample(n=8, seed=2).select("prob", "old", "new", "vocall").iter_rows():
        print("   ", r)
x.write_parquet("/work/lead/fr_contested_vocab.parquet")
