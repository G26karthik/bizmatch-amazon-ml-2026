"""Non-word swaps: typo of the S1's own word (copy) or typo of another vocabulary word (a sibling decoy's copy)?
For contested non-vocab swaps (fr_contested_vocab.parquet, voc1 <= 30): sim_old = similarity(new, replaced S1 word),
sim_alt = best similarity(new, any frequent French S1 vocabulary word other than the replaced one). Bracket share and
fr13 status by which is closer.

python scripts/fr13_typo_origin.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz, process

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["src", "country", "business_name"])
fr1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1))["business_name"].to_list()
voc1 = Counter(w for n in fr1 for w in set(re.findall(r"[a-z0-9]+", norm(n))))
vocab = [w for w, c in voc1.most_common(3000) if c > 30 and len(w) >= 3]
x = pl.read_parquet("/work/lead/fr_contested_vocab.parquet").filter(pl.col("voc1") <= 30)
restore = pl.read_parquet("/work/lead/fr13_restore.parquet").select("q", "s1").with_columns(pl.lit(True).alias("restored"))
x = x.join(restore, on=["q", "s1"], how="left").with_columns(pl.col("restored").fill_null(False))
so, sa, alt = [], [], []
for w, o in zip(x["new"].to_list(), x["old"].to_list()):
    so.append(fuzz.ratio(w, o))
    best = process.extract(w, vocab, scorer=fuzz.ratio, limit=2)
    b = next((c for c in best if c[0] != o), ("", 0, 0))
    sa.append(b[1]); alt.append(b[0])
x = x.with_columns(pl.Series("sim_old", so), pl.Series("sim_alt", sa), pl.Series("alt", alt))
x = x.with_columns(pl.when(pl.col("sim_alt") >= pl.col("sim_old") + 15).then(pl.lit("alt_much_closer"))
                   .when(pl.col("sim_alt") > pl.col("sim_old")).then(pl.lit("alt_closer"))
                   .otherwise(pl.lit("old_closer")).alias("origin"))
t = x.group_by("status", "restored", "origin").agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk")) \
     .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).sort("status", "restored", "origin")
print(t)
for o in ("alt_much_closer", "alt_closer"):
    print(f"\nexamples {o}")
    for r in x.filter(pl.col("origin") == o).sample(n=min(12, x.filter(pl.col("origin") == o).height), seed=4).select("status", "old", "new", "alt", "sim_old", "sim_alt").iter_rows():
        print("   ", r)
x.write_parquet("/work/lead/fr_typo_origin.parquet")
