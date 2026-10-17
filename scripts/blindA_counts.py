"""blindA: count structure, exact duplicates, train vs test ratios, submission sanity."""
import polars as pl, numpy as np
O = "/work/agent_blindA"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
pairs = pl.read_parquet(f"{O}/pairs_train.parquet")
s = {k: pl.read_parquet(f"{O}/raw_train_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}
t = {k: pl.read_parquet(f"{O}/raw_test_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}

print("== records per country, train vs test")
for nm, d in [("train", s), ("test", t)]:
    c = {k: d[k].group_by("country").len().rename({"len": f"S{k}"}) for k in [1, 2, 3]}
    x = c[1].join(c[2], on="country", how="full", coalesce=True).join(c[3], on="country", how="full", coalesce=True)
    x = x.with_columns(((pl.col("S2")) / pl.col("S1")).round(3).alias("S2/S1"), (pl.col("S3") / pl.col("S1")).round(3).alias("S3/S1"))
    print(nm); print(x.sort("country"))

print("== copies per S1 (train) by country")
s1c = s[1].select(pl.col("entity_id").alias("s1"), "country")
cp = pairs.group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
x = s1c.join(cp, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0))
x = x.with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
for ctry in ["US", "India"]:
    xx = x.filter(pl.col("country") == ctry)
    print(ctry, "total dist", xx.group_by("n").len().sort("n").with_columns((pl.col("len") / xx.height).round(4).alias("f")).to_dicts())
    print(ctry, "n2 dist", xx.group_by("n2").len().sort("n2").to_dicts())
    print(ctry, "n3 dist", xx.group_by("n3").len().sort("n3").to_dicts())
    j = xx.group_by(["n2", "n3"]).len().sort("len", descending=True).head(15)
    print(ctry, "joint top", j.to_dicts())
    # independence check: P(n2,n3) vs P(n2)P(n3)
    p2 = xx.group_by("n2").len().with_columns((pl.col("len") / xx.height).alias("p2")).select("n2", "p2")
    p3 = xx.group_by("n3").len().with_columns((pl.col("len") / xx.height).alias("p3")).select("n3", "p3")
    jj = xx.group_by(["n2", "n3"]).len().join(p2, on="n2").join(p3, on="n3").with_columns((pl.col("len") / xx.height / (pl.col("p2") * pl.col("p3"))).round(3).alias("lift")).sort("len", descending=True).head(12)
    print(ctry, "joint lift", jj.select("n2", "n3", "len", "lift").to_dicts())

print("== exact duplicate (name,addr) structure")
def key(df):
    return df.with_columns((pl.col("name") + "\x1f" + pl.col("addr")).alias("k"))
for nm, d in [("train", s), ("test", t)]:
    k1 = key(d[1]).select("k", pl.col("entity_id").alias("s1"))
    for k in [2, 3]:
        dk = key(d[k])
        dup_in_src = dk.group_by("k").len().filter(pl.col("len") > 1)
        ex_s1 = dk.join(k1, on="k", how="inner")
        print(f" {nm} S{k}: rows {dk.height}, keys with >1 rows {dup_in_src.height} (rows {dup_in_src['len'].sum()}), rows exactly equal an S1 (name+addr): {ex_s1['entity_id'].n_unique()}")
        if nm == "train":
            pm = pairs.filter(pl.col("src") == f"S{k}").select(pl.col("m").alias("entity_id"), "s1")
            e = ex_s1.join(pm, on="entity_id", how="left", suffix="_true")
            print(f"   exact-equal-S1 rows: truly matched to that S1: {(e['s1']==e['s1_true']).sum()}, matched to other S1: {((e['s1']!=e['s1_true']) & e['s1_true'].is_not_null()).sum()}, unmatched: {e['s1_true'].is_null().sum()}")
            dd = dk.join(dup_in_src.select("k"), on="k").join(pm, on="entity_id", how="left")
            g = dd.group_by("k").agg(pl.col("s1").n_unique().alias("nu"), pl.col("s1").null_count().alias("nnull"), pl.len())
            print(f"   within-source dup groups: all same s1 {(g['nu']==1).sum() - ((g['nnull']>0)&(g['nu']==1)).sum()}, mixed {(g['nu']>1).sum()}, groups with unmatched {(g['nnull']>0).sum()}")
    k1d = key(d[1]).group_by("k").len().filter(pl.col("len") > 1)
    print(f" {nm} S1 duplicate name+addr keys: {k1d.height}")
    n1d = d[1].group_by(["name", "country"]).len().filter(pl.col("len") > 1)
    print(f" {nm} S1 duplicate names (same country): keys {n1d.height} rows {n1d['len'].sum()}")

print("== S2<->S3 exact same (name,addr) across sources")
for nm, d in [("train", s), ("test", t)]:
    a = key(d[2]).select("k", pl.col("entity_id").alias("e2"))
    b = key(d[3]).select("k", pl.col("entity_id").alias("e3"))
    j = a.join(b, on="k")
    print(f" {nm}: S2-S3 exact equal pairs {j.height}")
    if nm == "train":
        pm = pairs.select(pl.col("m"), "s1")
        j = j.join(pm.rename({"m": "e2", "s1": "s1a"}), on="e2", how="left").join(pm.rename({"m": "e3", "s1": "s1b"}), on="e3", how="left")
        print("   same s1:", (j["s1a"] == j["s1b"]).sum(), "diff s1:", ((j["s1a"] != j["s1b"])).sum(), "one/both unmatched:", (j["s1a"].is_null() | j["s1b"].is_null()).sum())

print("== submission sanity")
sub = pl.read_csv("/work/out_fr14/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
print(" sub rows", sub.height, "S1 test", t[1].height, "cols", sub.columns)
sp = sub.with_columns(pl.col(sub.columns[1]).str.split(",")).explode(sub.columns[1]).filter(pl.col(sub.columns[1]) != "")
sp = sp.rename({sub.columns[0]: "s1", sub.columns[1]: "m"})
print(" sub pairs", sp.height, "m assigned to >1 S1:", sp.group_by("m").len().filter(pl.col("len") > 1).height)
sp = sp.with_columns(pl.col("m").str.slice(0, 2).alias("src"))
sp = sp.join(t[1].select(pl.col("entity_id").alias("s1"), "country"), on="s1")
for k in [2, 3]:
    tk = t[k].with_columns(pl.col("entity_id").is_in(sp["m"]).alias("assigned"))
    print(f" test S{k} assigned frac by country:", tk.group_by("country").agg(pl.col("assigned").mean(), pl.len()).sort("country").to_dicts())
for k in [2, 3]:
    sk = s[k].with_columns(pl.col("entity_id").is_in(pairs["m"]).alias("matched"))
    print(f" train S{k} matched frac by country:", sk.group_by("country").agg(pl.col("matched").mean(), pl.len()).sort("country").to_dicts())
sc = sp.group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
y = t[1].select(pl.col("entity_id").alias("s1"), "country").join(sc, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0)).with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
for ctry in ["US", "India", "France"]:
    yy = y.filter(pl.col("country") == ctry)
    print(" test sub", ctry, "total dist", yy.group_by("n").len().sort("n").with_columns((pl.col("len") / yy.height).round(4).alias("f")).to_dicts())
    print(" test sub", ctry, "n2 dist", yy.group_by("n2").len().sort("n2").to_dicts())
    print(" test sub", ctry, "n3 dist", yy.group_by("n3").len().sort("n3").to_dicts())
y.write_parquet(f"{O}/sub_counts.parquet")
