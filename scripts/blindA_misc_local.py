"""blindA misc checks (local): id collisions, GT list order, S1 attributes vs k, test dup consistency, test row/id corr."""
import polars as pl, numpy as np, sys, gc
R = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
def rd(f, cols=None):
    df = pl.read_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    df = df.with_row_index("row").with_columns(pl.col(df.columns[0]).str.slice(3).cast(pl.Int64).alias("idn")) if "entity_id" in df.columns else df
    return df.select(cols) if cols else df
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
pairs = pairs.with_columns(pl.col("m").str.slice(0, 2).alias("src"), pl.col("m").str.slice(3).cast(pl.Int64).alias("m_idn"), pl.int_range(pl.len()).over("s1").alias("pos"))
s1 = rd("train_source1.tsv")
print("== 1. numeric id collisions")
for k in [2, 3]:
    sk = rd(f"train_source{k}.tsv", ["entity_id", "idn"])
    j = s1.select("idn", pl.col("entity_id").alias("s1")).join(sk.rename({"entity_id": "m"}), on="idn")
    jj = j.join(pairs.select("s1", "m", pl.lit(True).alias("t")), on=["s1", "m"], how="left")
    print(f" S1-S{k} same numeric id: {j.height}; true pairs among them: {jj['t'].sum()} (chance ~{j.height*1.73/s1.height:.3f})")
    del sk; gc.collect()
print("== 2. GT list order within source")
for src in ["S2", "S3"]:
    g = pairs.filter(pl.col("src") == src).group_by("s1", maintain_order=True).agg(pl.col("m_idn"), pl.len().alias("n")).filter(pl.col("n") >= 2)
    g = g.with_columns(pl.col("m_idn").list.eval((pl.element().diff().drop_nulls() > 0).all()).list.first().alias("asc"))
    print(f" {src}: S1s with >=2 copies {g.height}; id-ascending lists {g['asc'].mean():.4f} (random n=2: 0.5)")
x = pairs.group_by("s1", maintain_order=True).agg(pl.col("src").str.join("").alias("sj"))
print(" lists with an S3 before an S2:", x.filter(pl.col("sj").str.contains("S3S2")).height, "of", x.height)
print("== 3. S1 attributes vs k")
k = pairs.group_by("s1").len().rename({"s1": "entity_id", "len": "k"})
f = s1.join(k, on="entity_id", how="left").with_columns(pl.col("k").fill_null(0))
f = f.with_columns((pl.col("k") == 0).alias("single"),
                   pl.col("business_name").str.split(" ").list.len().clip(1, 7).alias("nw"),
                   pl.col("business_name").str.contains(r"(?i)\b(inc|llc|ltd|corp|co|pvt|private|limited|llp|lp|pc|pllc|company|corporation|incorporated)\.?$").alias("legal"),
                   pl.col("business_address").str.count_matches(",").clip(0, 8).alias("ncomma"),
                   pl.col("business_address").str.contains(r"(?i)\bunit\b|\bapt\b|\bsuite\b|\bste\b|\bfl\b|floor").alias("unit"),
                   pl.col("business_address").str.contains(r"^\d").alias("starts_digit"),
                   pl.col("business_name").str.len_chars().clip(0, 60).floordiv(10).alias("nlen10"))
nd = s1.group_by(["business_name", "country"]).len().rename({"len": "name_dup"})
f = f.join(nd, on=["business_name", "country"], how="left").with_columns(pl.col("name_dup").clip(1, 6))
for c in ["country", "nw", "legal", "ncomma", "unit", "starts_digit", "nlen10", "name_dup"]:
    g = f.group_by(["country", c] if c != "country" else [c]).agg(pl.len(), pl.col("single").mean().round(4).alias("p_single"), pl.col("k").mean().round(3).alias("mean_k")).sort(["country", c] if c != "country" else [c])
    print(f" by {c}:", g.filter(pl.col("len") > 500).to_dicts())
del f, s1; gc.collect()
print("== 4/5. test: submission dup consistency and row/id correlation")
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"})
t1 = rd("test_source1.tsv", ["entity_id", "row", "idn"]).rename({"entity_id": "s1", "row": "r1", "idn": "i1"})
def rank(x):
    return np.argsort(np.argsort(x))
allrec = []
for kk in [2, 3]:
    tk = rd(f"test_source{kk}.tsv", ["entity_id", "business_name", "business_address", "country", "row", "idn"])
    q = sp.join(tk.select("entity_id", pl.col("row").alias("rm"), pl.col("idn").alias("im")), on="entity_id").join(t1, on="s1")
    print(f" S{kk}: n {q.height} spearman rows {np.corrcoef(rank(q['r1'].to_numpy()), rank(q['rm'].to_numpy()))[0,1]:.5f} ids {np.corrcoef(rank(q['i1'].to_numpy()), rank(q['im'].to_numpy()))[0,1]:.5f}")
    allrec.append(tk.select("entity_id", "business_name", "business_address", "country").with_columns(pl.lit(f"S{kk}").alias("src")))
a = pl.concat(allrec).join(sp, on="entity_id", how="left")
for scope, keys in [("same source", ["src", "business_name", "business_address"]), ("S2/S3 pooled", ["business_name", "business_address"])]:
    g = a.group_by(keys).agg(pl.len().alias("n"), pl.col("s1").n_unique().alias("nu"), pl.col("s1").null_count().alias("nn"), pl.col("country").first()).filter(pl.col("n") > 1)
    part = g.filter((pl.col("nn") > 0) & (pl.col("nn") < pl.col("n")))
    print(f" test exact-dup groups ({scope}): {g.height}; all unassigned {(g['nn']==g['n']).sum()}; partially assigned {part.height} (records unassigned in them {part['nn'].sum()}); split across S1s {(g.filter(pl.col('nn')==0)['nu']>1).sum()}")
    print("   partial by country:", part.group_by("country").len().to_dicts())
    if scope == "same source":
        part.write_parquet(sys.argv[3])
