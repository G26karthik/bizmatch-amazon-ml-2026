"""blindA: row-order and entity_id structure vs train truth."""
import polars as pl, numpy as np
O = "/work/agent_blindA"
rng = np.random.default_rng(0)
s = {k: pl.read_parquet(f"{O}/raw_train_s{k}.parquet") for k in [1, 2, 3]}
N = {k: s[k].height for k in s}
print("N", N)
gt = pl.read_parquet(f"{O}/gt_raw.parquet")
print("gt rows", gt.height, "S1 rows", N[1])
# GT order vs S1 file order
g1 = gt.join(s[1].select(["entity_id", "row"]), left_on="source1_entity_id", right_on="entity_id", how="left")
print("gt ids missing from S1:", g1["row"].null_count())
a = g1["gtrow"].to_numpy(); b = g1["row"].to_numpy()
print("spearman gtrow vs s1row:", np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1],
      "frac identical order:", float((a == b).mean()))
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
         .with_columns(pl.col("m").str.slice(0, 2).alias("src"),
                       pl.col("matched_entity_ids" if False else "m").str.slice(3).cast(pl.Int64).alias("m_idn")))
# position of m in GT list
pairs = pairs.with_columns(pl.int_range(pl.len()).over("s1").alias("pos_in_list"))
print("pairs", pairs.height, pairs.group_by("src").len().sort("src"))
pairs = pairs.join(s[1].select(pl.col("entity_id").alias("s1"), pl.col("row").alias("s1_row"), pl.col("idn").alias("s1_idn")), on="s1")
parts = []
for k, src in [(2, "S2"), (3, "S3")]:
    p = pairs.filter(pl.col("src") == src).join(s[k].select(pl.col("entity_id").alias("m"), pl.col("row").alias("m_row")), on="m", how="left")
    print(src, "pairs", p.height, "m missing from file", p["m_row"].null_count())
    parts.append(p)
pairs = pl.concat(parts)
pairs.write_parquet(f"{O}/pairs_train.parquet")

def rank(x):
    return np.argsort(np.argsort(x))

for src, k in [("S2", 2), ("S3", 3)]:
    p = pairs.filter(pl.col("src") == src)
    x = p["s1_row"].to_numpy() / N[1]; y = p["m_row"].to_numpy() / N[k]
    print(f"\n== {src}: spearman s1_row vs m_row = {np.corrcoef(rank(x), rank(y))[0,1]:.5f}")
    d = y - x
    print("  rel pos diff quantiles", np.quantile(d, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(4))
    print("  frac |diff|<0.001:", float((np.abs(d) < 0.001).mean()), " random expect ~0.002")
    xi = p["s1_idn"].to_numpy().astype(float); yi = p["m_idn"].to_numpy().astype(float)
    print(f"  spearman s1_idn vs m_idn = {np.corrcoef(rank(xi), rank(yi))[0,1]:.5f}")
    for mod in [2, 3, 5, 7, 10, 11, 13, 100, 1000]:
        eq = float((p["s1_idn"].to_numpy() % mod == p["m_idn"].to_numpy() % mod).mean())
        print(f"   mod {mod}: eq frac {eq:.4f} (chance {1/mod:.4f})")
    # idn vs row within file correlation
    print(f"  spearman m_row vs m_idn (file) :", np.corrcoef(rank(s[k]['row'].to_numpy()), rank(s[k]['idn'].to_numpy()))[0, 1])
    # consecutive copies of same s1 within this source: row gaps
    g = p.group_by("s1").agg(pl.col("m_row").sort().diff().drop_nulls().alias("gaps"), pl.len().alias("n"))
    gaps = g.explode("gaps")["gaps"].drop_nulls().to_numpy()
    print(f"  within-S1 same-source copy row gaps: n={len(gaps)} median={np.median(gaps) if len(gaps) else None}, frac<=10: {(gaps<=10).mean() if len(gaps) else None}, frac<=1000: {(gaps<=1000).mean() if len(gaps) else None}; random median ~{N[k]/3:.0f}")
    gi = p.group_by("s1").agg(pl.col("m_idn").sort().diff().drop_nulls().alias("gaps")).explode("gaps")["gaps"].drop_nulls().to_numpy()
    print(f"  within-S1 same-source copy idn gaps: median={np.median(gi)}, frac<=1000: {(gi<=1000).mean()}, frac<=1e6 {(gi<=1e6).mean()}")
    # position in GT list vs m_row
    print("  mean m_row rel by pos_in_list:", p.group_by("pos_in_list").agg((pl.col("m_row").mean() / N[k]).alias("r"), pl.len()).sort("pos_in_list").head(8).to_dicts())

print("\n== S2 vs S3 copies of same S1: row and idn relation")
p2 = pairs.filter(pl.col("src") == "S2").select("s1", pl.col("m_row").alias("r2"), pl.col("m_idn").alias("i2"))
p3 = pairs.filter(pl.col("src") == "S3").select("s1", pl.col("m_row").alias("r3"), pl.col("m_idn").alias("i3"))
x = p2.join(p3, on="s1")
print(" cross pairs", x.height, "spearman r2 r3", np.corrcoef(rank(x["r2"].to_numpy()), rank(x["r3"].to_numpy()))[0, 1],
      "spearman i2 i3", np.corrcoef(rank(x["i2"].to_numpy()), rank(x["i3"].to_numpy()))[0, 1],
      "frac i2==i3", float((x["i2"] == x["i3"]).mean()))

print("\n== matched vs unmatched by file position decile / idn decile")
matched = set(pairs["m"].to_list())
for k in [2, 3]:
    df = s[k].with_columns(pl.col("entity_id").is_in(pairs["m"]).alias("matched"))
    df = df.with_columns((pl.col("row") * 10 // df.height).alias("rdec"),
                         (pl.col("idn").rank() * 10 // (df.height + 1)).cast(pl.Int64).alias("idec"),
                         pl.col("idn").cast(pl.Utf8).str.len_chars().alias("ndig"))
    print(f" S{k} matched frac overall {df['matched'].mean():.4f}")
    print("  by row decile:", df.group_by("rdec").agg(pl.col("matched").mean()).sort("rdec")["matched"].round(4).to_list())
    print("  by idn decile:", df.group_by("idec").agg(pl.col("matched").mean()).sort("idec")["matched"].round(4).to_list())
    print("  by idn ndigits:", df.group_by("ndig").agg(pl.col("matched").mean(), pl.len()).sort("ndig").to_dicts())
    print("  by country:", df.group_by("country").agg(pl.col("matched").mean(), pl.len()).sort("country").to_dicts())
    # runs: is matched status autocorrelated in file order?
    m = df.sort("row")["matched"].to_numpy().astype(int)
    print("  P(matched|prev matched)", m[1:][m[:-1] == 1].mean(), "P(matched|prev unmatched)", m[1:][m[:-1] == 0].mean())
    # country runs in file order
    c = df.sort("row")["country"].to_numpy()
    print("  P(same country as prev row)", float((c[1:] == c[:-1]).mean()))
s1 = s[1].join(gt.select(pl.col("source1_entity_id").alias("entity_id"), (pl.col("matched_entity_ids") == "").alias("single")), on="entity_id")
print(" S1 singleton frac", s1["single"].mean(), "by row decile", s1.with_columns((pl.col("row") * 10 // s1.height).alias("d")).group_by("d").agg(pl.col("single").mean()).sort("d")["single"].round(4).to_list())
print(" S1 singleton by idn ndig", s1.with_columns(pl.col("idn").cast(pl.Utf8).str.len_chars().alias("nd")).group_by("nd").agg(pl.col("single").mean(), pl.len()).sort("nd").to_dicts())
# ids reused across sources?
all_ids = pl.concat([s[k].select("idn", pl.lit(k).alias("k")) for k in [1, 2, 3]])
print(" idn shared across sources:", all_ids.group_by("idn").agg(pl.col("k").n_unique().alias("nk")).filter(pl.col("nk") > 1).height)
t = {k: pl.read_parquet(f"{O}/raw_test_s{k}.parquet") for k in [1, 2, 3]}
all_ids2 = pl.concat([t[k].select("idn", pl.lit(k).alias("k")) for k in [1, 2, 3]] + [s[k].select("idn", pl.lit(10 + k).alias("k")) for k in [1, 2, 3]])
print(" idn shared across all train+test files:", all_ids2.group_by("idn").agg(pl.col("k").n_unique().alias("nk")).filter(pl.col("nk") > 1).height)
for k in [1, 2, 3]:
    c = t[k].sort("row")["country"].to_numpy()
    print(f" test S{k} P(same country as prev)", float((c[1:] == c[:-1]).mean()), t[k].group_by("country").len().sort("country").to_dicts())
