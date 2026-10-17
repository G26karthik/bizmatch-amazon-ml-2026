"""Bracket-share decoy estimate by subclass: fr_swap_subsets categories (A exact type, B near vocab, T near typo, C street-typo
type, D street-typo generic, E generic) x fr12 status x whether the S1 has other fr12 matches; plus the removed set split
by legal/lowercase. d = (0.0418 - brk) / (0.0418 - 0.0090) (copy ref generic|same, decoy ref type|offset; bracket_mix.py).

python scripts/bracket_split.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list()
eid = rec.select("rid", "entity_id")


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
            .filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


a12, a7 = accepted("/work/out_fr12"), accepted("/work/out_fr7")
ss = pl.read_parquet("/work/dive/fr_swap_subsets.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "cat").unique(["q", "s1"])
br, lo = [], []
for q, s in zip(ss["q"].to_list(), ss["s1"].to_list()):
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
    w = next(iter(A - B)) if len(A - B) == 1 else None
    br.append(bool(w) and bool(re.search(r"[\[\(]\s*" + re.escape(w), norm(nm[q]))))
    lo.append(nm[q] == nm[q].lower())
ss = ss.with_columns(pl.Series("bracket", br), pl.Series("lower", lo))
ss = ss.join(a12.with_columns(pl.lit(True).alias("k12")), on=["q", "s1"], how="left").join(a7.with_columns(pl.lit(True).alias("k7")), on=["q", "s1"], how="left") \
       .with_columns(pl.col("k12").fill_null(False), pl.col("k7").fill_null(False))
n12 = a12.group_by("s1").len().rename({"len": "n12"})
ss = ss.join(n12, on="s1", how="left").with_columns(pl.col("n12").fill_null(0))
ss = ss.with_columns(pl.when(pl.col("k12")).then(pl.lit("kept")).when(pl.col("k7")).then(pl.lit("removed")).otherwise(pl.lit("rejected")).alias("status"),
                     pl.when(pl.col("k12")).then(pl.col("n12") > 1).otherwise(pl.col("n12") > 0).alias("s1_has_other"))
t = ss.group_by("cat", "status", "s1_has_other").agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk"),
                                                    pl.col("lower").mean().round(4).alias("low")) \
      .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).filter(pl.col("n") >= 40).sort("cat", "status", "s1_has_other")
print(t)
ss.write_parquet("/work/lead/fr_swap_brk.parquet")
