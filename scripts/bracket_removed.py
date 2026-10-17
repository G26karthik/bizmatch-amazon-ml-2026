"""Split the removed / kept French swap classes (fr_swap_brk.parquet from bracket_split.py) by model probability, source,
the S1's other-match count, old type word and near-typo vocabulary presence; bracket-share decoy estimate per slice.
d = (0.0418 - brk) / (0.0418 - 0.0090); a restore is worth it only where d < ~0.27.

python scripts/bracket_removed.py
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


x = pl.read_parquet("/work/lead/fr_swap_brk.parquet")
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name", "src", "country"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); srcl = rec["src"].to_list()
fr1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1))["business_name"].to_list()
voc = Counter(w for n in fr1 for w in set(re.findall(r"[a-z0-9]+", norm(n))) - LEG)
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
x = x.join(tb, on=["q", "s1"], how="left")
old, new = [], []
for q, s in zip(x["q"].to_list(), x["s1"].to_list()):
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
    old.append(next(iter(B - A)) if len(B - A) == 1 else "?"); new.append(next(iter(A - B)) if len(A - B) == 1 else "?")
x = x.with_columns(pl.Series("old", old), pl.Series("new", new), pl.Series("src", [srcl[q] for q in x["q"].to_list()]),
                   pl.Series("new_voc", [voc.get(w, 0) for w in new]))
x = x.with_columns(pl.col("prob").fill_null(-1).cut([0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]).alias("pbin"),
                   pl.col("n12").cut([1, 2, 3, 4]).alias("n12bin"),
                   pl.col("new_voc").cut([0, 5, 50, 500]).alias("vocbin"))


def show(df, by, title):
    t = df.group_by(by).agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk")) \
          .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk"),
                        # 90% interval on d from a normal approx of the bracket rate
                        ((BC - (pl.col("brk") + 1.645 * (pl.col("brk").clip(0.005) * (1 - pl.col("brk")) / pl.col("n")).sqrt())) / (BC - BD)).round(2).alias("d_lo"),
                        ((BC - (pl.col("brk") - 1.645 * (pl.col("brk").clip(0.005) * (1 - pl.col("brk")) / pl.col("n")).sqrt())) / (BC - BD)).round(2).alias("d_hi")) \
          .sort(by)
    print(f"\n=== {title}"); print(t)


rem = x.filter(pl.col("cat").is_in(["A_exact_type", "C_edit_type"]) & (pl.col("status") == "removed"))
show(rem, ["pbin"], "removed type swaps by blended prob")
show(rem, ["src"], "removed type swaps by source")
show(rem, ["n12bin"], "removed type swaps by S1's fr12 match count")
top = rem.group_by("old").len().sort("len", descending=True).head(12)["old"].to_list()
show(rem.filter(pl.col("old").is_in(top)), ["old"], "removed type swaps by replaced S1 word (top 12)")
kt = x.filter((pl.col("cat") == "T_exact_near_typo") & (pl.col("status") == "kept"))
show(kt, ["vocbin"], "kept near-typo swaps by vocabulary count of the new word")
show(kt, ["pbin"], "kept near-typo swaps by blended prob")
