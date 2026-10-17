"""Summarise g1 parts: name/address noise frequencies and combinations by country x source (true pairs sample)."""
import collections
import glob
import pickle

import polars as pl

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(75)
df = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob("/work/dive/g1v2_pairs_*.parquet"))])
ndet, adet = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for p in glob.glob("/work/dive/g1v2_det_*.pkl"):
    nd, ad = pickle.load(open(p, "rb"))
    for k, v in nd.items():
        ndet[k].update(v)
    for k, v in ad.items():
        adet[k].update(v)
print("pairs", df.height, df.group_by("country", "src").len().sort("country", "src").to_dicts())
# collapse case-pattern ops into one flag for the combination view
df = df.with_columns(
    pl.col("nops").str.split("|").list.eval(pl.element().str.replace(r"^case:.*", "case_change")).list.unique().list.sort().list.join("|").alias("ncombo"),
    pl.col("aops").str.split("|").list.eval(pl.element().str.replace(r"^case:.*", "case_change")).list.unique().list.sort().list.join("|").alias("acombo"))


def share(col, explode=True, top=40, min_pct=0.05):
    d = df.select("country", "src", pl.col(col).str.split("|").alias("f"))
    d = d.explode("f") if explode else df.select("country", "src", pl.col(col).alias("f"))
    tot = df.group_by("country", "src").len().rename({"len": "tot"})
    t = d.group_by("country", "src", "f").len().join(tot, on=["country", "src"]) \
         .with_columns((100 * pl.col("len") / pl.col("tot")).round(2).alias("pct"))
    w = t.with_columns(pl.format("{}-S{}", "country", "src").alias("k")).pivot(on="k", index="f", values="pct").fill_null(0)
    cols = [c for c in ["US-S2", "US-S3", "India-S2", "India-S3"] if c in w.columns]
    w = w.with_columns(pl.max_horizontal(cols).alias("mx")).filter(pl.col("mx") >= min_pct).drop("mx")
    print(w.select(["f"] + cols).sort(pl.sum_horizontal(cols), descending=True).head(top))


print("\n=== NAME op flags (% of true pairs; multi-label) ===")
share("nops", top=50)
print("\n=== NAME op COMBINATIONS (case patterns merged) ===")
share("ncombo", explode=False, top=40)
print("\n=== ADDRESS op flags ===")
share("aops", top=50)
print("\n=== ADDRESS op COMBINATIONS ===")
share("acombo", explode=False, top=40)
print("\n=== HOUSE change kind ===")
share("house_k", explode=False)
print("\n=== POSTAL change kind ===")
share("postal_k", explode=False)
print("\n=== n name ops per pair (excluding case) ===")
print(df.with_columns(pl.col("ncombo").str.split("|").list.eval(pl.element().filter(~pl.element().is_in(["case_change", "exact", "case_only"]))).list.len().alias("k"))
      .group_by("country", "src", "k").len().sort("country", "src", "k").pivot(on="k", index=["country", "src"], values="len"))
for (ctry, k), cnt in sorted(ndet.items()):
    if sum(cnt.values()) >= 200:
        print(f"NAME {ctry} {k:14} n={sum(cnt.values()):7d} top: {cnt.most_common(18)}")
for (ctry, k), cnt in sorted(adet.items()):
    if sum(cnt.values()) >= 200:
        print(f"ADDR {ctry} {k:14} n={sum(cnt.values()):7d} top: {cnt.most_common(18)}")
