"""blindA misc checks:
 1. numeric-id collisions S1 vs S2/S3: true-pair rate vs chance
 2. GT list order within source (sorted by id? by row?)
 3. S2-S3 copies of the same S1: shared raw noise features vs independence
 4. S1 singleton / copy-count k vs raw S1 features
 5. within-source exact duplicates: consistency of test submission
 6. test: row / id correlation within submitted pairs
"""
import polars as pl, numpy as np
O = "/work/agent_blindA"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
pairs = pl.read_parquet(f"{O}/pairs_train.parquet")
gt = pl.read_parquet(f"{O}/gt_raw.parquet")
s = {k: pl.read_parquet(f"{O}/raw_train_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}
t = {k: pl.read_parquet(f"{O}/raw_test_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}

print("== 1. numeric id collisions")
for k in [2, 3]:
    j = s[1].select(pl.col("idn"), pl.col("entity_id").alias("s1")).join(s[k].select("idn", pl.col("entity_id").alias("m")), on="idn")
    jj = j.join(pairs.select("s1", "m", pl.lit(True).alias("true")), on=["s1", "m"], how="left")
    print(f" S1-S{k} same numeric id: {j.height} pairs, true pairs among them {jj['true'].sum()} (chance ~{j.height * 3.46 / 2 / s[1].height:.4f})")

print("== 2. GT list order")
p = pairs.with_columns(pl.int_range(pl.len()).over(["s1", "src"]).alias("pos_src"))
for src in ["S2", "S3"]:
    g = p.filter(pl.col("src") == src).group_by("s1").agg(pl.col("m_idn"), pl.col("m_row"), pl.len().alias("n")).filter(pl.col("n") >= 2)
    si = g.with_columns(pl.col("m_idn").list.eval(pl.element().diff().drop_nulls() > 0).list.all().alias("asc"),
                        pl.col("m_idn").list.eval(pl.element().diff().drop_nulls() < 0).list.all().alias("desc"),
                        pl.col("m_row").list.eval(pl.element().diff().drop_nulls() > 0).list.all().alias("rasc"))
    print(f" {src}: S1s with >=2 copies {g.height}; lists id-ascending {si['asc'].mean():.4f} id-desc {si['desc'].mean():.4f} row-asc {si['rasc'].mean():.4f} (random for n=2: 0.5)")
# interleaving: are S2 always before S3 in the list?
p2 = pairs.with_columns(pl.int_range(pl.len()).over("s1").alias("pos"))
x = p2.group_by("s1").agg(pl.col("src").alias("srcs"))
x = x.with_columns(pl.col("srcs").list.join("").alias("sj"))
print(" lists with S3 before an S2:", x.filter(pl.col("sj").str.contains("S3S2")).height, "of", x.height)

print("== 3. shared noise between S2 and S3 copies of same S1 (raw features)")
fm = {k: pl.read_parquet(f"{O}/fmt_train_s{k}.parquet") for k in [2, 3]}
feats = ["n_dblspace", "n_nonascii", "n_alllower", "n_allupper", "n_lead_punct", "n_trail_punct", "n_bracket", "n_web", "a_empty", "a_digit", "n_odd_char", "a_lead_punct", "n_indic", "a_indic", "n_latin_ext"]
a = pairs.filter(pl.col("src") == "S2").select("s1", pl.col("m").alias("entity_id")).join(fm[2].select(["entity_id"] + feats), on="entity_id")
b = pairs.filter(pl.col("src") == "S3").select("s1", pl.col("m").alias("entity_id")).join(fm[3].select(["entity_id"] + feats), on="entity_id")
# one random S2 copy and one random S3 copy per S1
a1 = a.group_by("s1").first(); b1 = b.group_by("s1").first()
ab = a1.join(b1, on="s1", suffix="_3")
for f in feats:
    pa = ab[f].mean(); pb = ab[f + "_3"].mean(); pj = (ab[f] & ab[f + "_3"]).mean()
    print(f"  {f:14s} P2 {pa:.4f} P3 {pb:.4f} P(both) {pj:.5f} indep {pa*pb:.5f} lift {pj/(pa*pb+1e-12):.2f}")
# within-source: two S2 copies of same S1
a2 = a.with_columns(pl.int_range(pl.len()).over("s1").alias("i")).filter(pl.col("i") < 2)
a2 = a2.filter(pl.col("i") == 0).join(a2.filter(pl.col("i") == 1), on="s1", suffix="_b")
print(" two S2 copies of same S1:")
for f in feats:
    pa = a2[f].mean(); pb = a2[f + "_b"].mean(); pj = (a2[f] & a2[f + "_b"]).mean()
    print(f"  {f:14s} P {pa:.4f} {pb:.4f} P(both) {pj:.5f} indep {pa*pb:.5f} lift {pj/(pa*pb+1e-12):.2f}")

print("== 4. S1 raw features vs singleton / k")
f1 = pl.read_parquet(f"{O}/fmt_train_s1.parquet")
kk = pairs.group_by("s1").len().rename({"s1": "entity_id", "len": "k"})
f1 = f1.join(kk, on="entity_id", how="left").with_columns(pl.col("k").fill_null(0))
f1 = f1.join(s[1].select("entity_id", "name", "addr"), on="entity_id")
f1 = f1.with_columns((pl.col("k") == 0).alias("single"), pl.col("name").str.split(" ").list.len().alias("nw"),
                     pl.col("name").str.contains(r"(?i)\b(inc|llc|ltd|corp|co|pvt|private|limited|llp|lp|pc|pllc|company|corporation)\.?$").alias("legal"))
nd = s[1].group_by(["name", "country"]).len().rename({"len": "name_dup"})
f1 = f1.join(nd, on=["name", "country"], how="left")
ad = s[1].group_by(["addr", "country"]).len().rename({"len": "addr_dup"})
f1 = f1.join(ad, on=["addr", "country"], how="left")
for c in ["country", "nw", "legal", "n_trail_punct", "n_lead_punct", "n_bracket", "n_digit", "a_digit", "name_dup", "addr_dup", "a_ncomma"]:
    g = f1.group_by(c).agg(pl.col("single").mean().round(4).alias("p_single"), pl.col("k").mean().round(3).alias("mean_k"), pl.len()).sort(c)
    print(f" by {c}:", g.filter(pl.col("len") > 200).to_dicts()[:14])
f1.select("entity_id", "k", "single", "nw", "legal", "name_dup", "addr_dup").write_parquet(f"{O}/s1_k_feats.parquet")

print("== 5. within-source exact duplicates in test: submission consistency")
sub = pl.read_csv("/work/out_fr14/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"})
allm = pl.concat([t[2], t[3]]).join(sp, on="entity_id", how="left")
for scope, keys in [("same source", ["src_", "name", "addr"]), ("across S2/S3", ["name", "addr"])]:
    d = allm.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src_"))
    g = d.group_by(keys).agg(pl.len().alias("n"), pl.col("s1").n_unique().alias("nu"), pl.col("s1").null_count().alias("nn"), pl.col("country").first()).filter(pl.col("n") > 1)
    print(f" test dup groups {scope}: {g.height}; all unassigned {(g['nn']==g['n']).sum()}; all same S1 {((g['nn']==0)&(g['nu']==1)).sum()}; partially assigned {((g['nn']>0)&(g['nn']<g['n'])).sum()}; different S1s {(g.filter(pl.col('nn')==0)['nu']>1).sum()}")
    print("   partially assigned by country:", g.filter((pl.col("nn") > 0) & (pl.col("nn") < pl.col("n"))).group_by("country").len().to_dicts())
# train analog: same-source dup groups -> truth
for k in [2, 3]:
    d = s[k].join(pairs.select(pl.col("m").alias("entity_id"), "s1"), on="entity_id", how="left")
    g = d.group_by(["name", "addr"]).agg(pl.len().alias("n"), pl.col("s1").n_unique().alias("nu"), pl.col("s1").null_count().alias("nn")).filter(pl.col("n") > 1)
    print(f" train S{k} dup groups: {g.height}; all unmatched {(g['nn']==g['n']).sum()}; all same S1 {((g['nn']==0)&(g['nu']==1)).sum()}; partial {((g['nn']>0)&(g['nn']<g['n'])).sum()}; diff S1 {(g.filter(pl.col('nn')==0)['nu']>1).sum()}")

print("== 6. test submitted pairs: row/id correlation")
def rank(x):
    return np.argsort(np.argsort(x))
t1 = t[1].select(pl.col("entity_id").alias("s1"), pl.col("row").alias("r1"), pl.col("idn").alias("i1"))
for k in [2, 3]:
    q = sp.join(t[k].select("entity_id", pl.col("row").alias("rm"), pl.col("idn").alias("im")), on="entity_id").join(t1, on="s1")
    print(f" S{k}: n {q.height} spearman rows {np.corrcoef(rank(q['r1'].to_numpy()), rank(q['rm'].to_numpy()))[0,1]:.5f} ids {np.corrcoef(rank(q['i1'].to_numpy()), rank(q['im'].to_numpy()))[0,1]:.5f}")
