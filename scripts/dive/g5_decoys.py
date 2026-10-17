"""Generator dive 5 (worker): decoy anatomy + rule precision over ALL S2/S3 records vs their best full-pool S1 candidate.
cls: true = best candidate is the true parent; other = true copy of a different S1; decoy = raw-unmatched record.
needs allq_{US,India}.parquet, best2.parquet (g2_prep.py)
"""
import sys

import polars as pl

pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(60)
C = sys.argv[1]
b = pl.read_parquet("/work/dive/best2.parquet").filter(pl.col("qc") == C)
print(f"########## {C}: S2/S3 {b.height}, decoys {(~b['matched']).sum()}, decoys w/o any candidate "
      f"{(~b['matched'] & b['s1'].is_null()).sum()}, copies w/o candidate {(b['matched'] & b['s1'].is_null()).sum()}")
print("best-candidate pA bins by class:")
print(b.filter(pl.col("s1").is_not_null()).with_columns(
    pl.when(pl.col("y")).then(pl.lit("true")).when(pl.col("matched")).then(pl.lit("other")).otherwise(pl.lit("decoy")).alias("cls"),
    pl.col("pA").cut([0.01, 0.1, 0.5, 0.9, 0.99]).alias("pAbin")).group_by("cls", "pAbin").len().sort("cls", "pAbin")
      .pivot(on="cls", index="pAbin", values="len"))
del b

LEGRE = r"(?i)\b(private limited|pvt\.? ?ltd\.?|pvt|limited|ltd\.?|l\.?l\.?c\.?|inc\.?|incorporated|corp\.?|corporation|co\.|company|llp|p\.?c\.?|pllc|lp|enterprises|associates|group|partners)\s*[\]\)]?\s*$"
FRAC = 3  # use 30% of records (hash on record id) to fit the CPU budget; counts below are x1/0.3 of the population
a = pl.scan_parquet(f"/work/dive/allq_{C}.parquet").filter(pl.col("mr").hash(3) % 10 < FRAC).collect()
print("sampled rows", a.height)
a = a.with_columns(
    pl.when(pl.col("y")).then(pl.lit("true")).when(pl.col("matched")).then(pl.lit("other")).otherwise(pl.lit("decoy")).alias("cls"),
    pl.col("name_n_2").str.split(" ").list.unique().alias("qt"), pl.col("name_n_1").str.split(" ").list.unique().alias("st"),
    pl.col("house_2").cast(pl.Int64, strict=False).alias("h2"), pl.col("house_1").cast(pl.Int64, strict=False).alias("h1"),
)
a = a.with_columns(pl.col("qt").list.set_difference("st").alias("extra"), pl.col("st").list.set_difference("qt").alias("miss"))
a = a.with_columns(
    pl.when(pl.col("name_n_2") == pl.col("name_n_1")).then(pl.lit("same"))
      .when((pl.col("extra").list.len() > 0) & (pl.col("miss").list.len() == 0)).then(pl.lit("s1+extra"))
      .when((pl.col("extra").list.len() == 0) & (pl.col("miss").list.len() == 0)).then(pl.lit("same_set"))
      .when((pl.col("extra").list.len() == 0)).then(pl.lit("s1-missing"))
      .when(pl.col("qt").list.set_intersection("st").list.len() == 0).then(pl.lit("disjoint"))
      .otherwise(pl.lit("replace")).alias("name_rel"),
    pl.when(pl.col("addr_empty_2")).then(pl.lit("q_empty"))
      .when(pl.col("addr_empty_1")).then(pl.lit("s1_empty"))
      .when(pl.col("addr_w_2") == pl.col("addr_w_1")).then(pl.lit("same_words"))
      .when(pl.col("street_tsr") >= 90).then(pl.lit("street~"))
      .otherwise(pl.lit("other_street")).alias("addr_rel"),
    pl.when(pl.col("addr_empty_2") | pl.col("addr_empty_1")).then(pl.lit("na"))
      .when((pl.col("house_2") == "") | (pl.col("house_1") == "")).then(pl.lit("missing"))
      .when(pl.col("house_2") == pl.col("house_1")).then(pl.lit("0"))
      .when(pl.col("h1").is_null() | pl.col("h2").is_null()).then(pl.lit("alnum"))
      .otherwise(pl.when((pl.col("h2") - pl.col("h1")).abs() <= 2).then(pl.lit("1-2"))
                 .when((pl.col("h2") - pl.col("h1")).abs() <= 10).then(pl.lit("3-10"))
                 .when((pl.col("h2") - pl.col("h1")).abs() <= 50).then(pl.lit("11-50"))
                 .when((pl.col("h2") - pl.col("h1")).abs() <= 1000).then(pl.lit("51-1000"))
                 .otherwise(pl.lit(">1000"))).alias("hdelta"),
    pl.col("business_name_2").str.extract(LEGRE, 1).str.to_lowercase().str.replace_all(r"[^a-z ]", "").fill_null("").alias("leg2"),
    pl.col("business_name_1").str.extract(LEGRE, 1).str.to_lowercase().str.replace_all(r"[^a-z ]", "").fill_null("").alias("leg1"),
)


def tab(keys, df=a, min_n=200, top=60, sort="n"):
    t = df.group_by(keys).agg(pl.len().alias("n"), (pl.col("cls") == "true").mean().round(4).alias("p_true"),
                              (pl.col("cls") == "decoy").mean().round(4).alias("p_decoy"),
                              (pl.col("cls") == "other").mean().round(4).alias("p_other"))
    print(t.filter(pl.col("n") >= min_n).sort(sort, descending=True).head(top))


print("\n=== name relation x class (row = best-candidate relation) ===")
tab(["name_rel"])
print("\n=== address relation ===")
tab(["addr_rel"])
print("\n=== house delta (non-empty both) ===")
tab(["hdelta"])
print("\n=== name_rel x addr_rel x hdelta (top combos) ===")
tab(["name_rel", "addr_rel", "hdelta"], top=45)
print("\n=== by source: name_rel x hdelta ===")
tab(["src", "name_rel", "hdelta"], top=40)
# extra words (added relative to nearest S1): count and precision per word
ex = a.select("cls", "src", "name_rel", "addr_rel", "hdelta", pl.col("extra")).filter(pl.col("extra").list.len() > 0) \
      .with_columns(pl.col("extra").list.len().alias("n_extra")).explode("extra")
print("\n=== extra word (vs nearest S1) -- most common overall, with share true/decoy ===")
tab(["extra"], df=ex, min_n=300, top=60)
print("\n=== extra words that are strong DECOY markers (n>=200, p_true<0.1) ===")
t = ex.group_by("extra").agg(pl.len().alias("n"), (pl.col("cls") == "true").mean().round(4).alias("p_true"),
                             (pl.col("cls") == "decoy").mean().round(4).alias("p_decoy"),
                             (pl.col("hdelta") == "0").mean().round(3).alias("house_same"),
                             (pl.col("name_rel") == "s1+extra").mean().round(3).alias("pure_add"))
print(t.filter((pl.col("n") >= 200) & (pl.col("p_true") < 0.1)).sort("n", descending=True).head(60))
print("\n=== extra words that are strong TRUE markers (n>=300, p_true>0.8) ===")
print(t.filter((pl.col("n") >= 300) & (pl.col("p_true") > 0.8)).sort("n", descending=True).head(40))
# missing words
mi = a.select("cls", pl.col("miss")).filter(pl.col("miss").list.len() > 0).explode("miss")
print("\n=== words dropped from the S1 name (n>=300) ===")
tab(["miss"], df=mi, min_n=300, top=30)
print("\n=== legal suffix transitions S1 -> record (raw names) ===")
tab(["leg1", "leg2"], min_n=500, top=50)
# decoy anatomy restricted to decoys
d = a.filter(pl.col("cls") == "decoy")
print(f"\n=== DECOYS ({d.height}) name_rel x addr_rel ===")
print(d.group_by("name_rel", "addr_rel").len().with_columns((pl.col("len") / d.height).round(4).alias("share")).sort("len", descending=True).head(30))
print("decoys: hdelta distribution (non-empty):")
print(d.group_by("hdelta").len().with_columns((pl.col("len") / d.height).round(4).alias("share")).sort("len", descending=True))
print("decoys: n extra words", d.select(pl.col("extra").list.len().alias("k")).group_by("k").len().sort("k").head(8))
print("decoys pA>=0.5 share", (d["pA"] >= 0.5).mean(), " pA>=0.9", (d["pA"] >= 0.9).mean())
