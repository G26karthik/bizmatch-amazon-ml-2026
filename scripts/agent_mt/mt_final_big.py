"""agent_mt BIG final: production-residual check split by 'scored' (record's best pair in the production uncertain zone,
local_data/llm/unc_{val,test}.parquet) and 'agree' (production's best S1 == our pick); apply to test with the chosen filter.

uv run --no-project --with polars==1.44.2 --with lightgbm --with numpy python scripts/agent_mt/mt_final_big.py [--write]
"""
import sys

import lightgbm as lgb  # noqa: F401
import polars as pl

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
W = "local_data/work/agent_mt"
SEED, VAL_MOD, HIDE = 42, 10, 19


def ridmap(split):
    return pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet").select("entity_id").with_columns(pl.lit(s).cast(pl.Int8).alias("src_n"))
                      for s in (1, 2, 3)]).with_row_index("rid").collect()


def unc(split, rec):
    u = pl.read_parquet(f"local_data/llm/unc_{split}.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("prob").cast(pl.Float64))
    u = u.join(rec.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("u_s1")), on="s1").join(rec.select(pl.col("rid").alias("q"), pl.col("entity_id").alias("m")), on="q")
    return u.group_by("m").agg(pl.col("prob").max().alias("pmax"), pl.col("u_s1").sort_by("prob").last())


# ---- val
rec = ridmap("train")
rec = rec.with_columns(hidden=(pl.col("src_n") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE), own_val=pl.col("entity_id").hash(SEED) % VAL_MOD == 0)
pk = pl.read_parquet(f"{W}/oof_big_picks.parquet")
tp = pl.read_parquet("local_data/train_pairs.parquet").join(pk.select("m"), on="m")
s1i = rec.filter(pl.col("src_n") == 1).select(pl.col("entity_id").alias("owner"), pl.col("hidden").alias("own_hidden"), pl.col("own_val").alias("par_val"))
v = pk.join(tp.rename({"s1": "owner"}), on="m", how="left").join(s1i, on="owner", how="left")
v = v.join(rec.select(pl.col("entity_id").alias("m"), pl.col("own_val").alias("m_val")), on="m", how="left")
v = v.with_columns(is_val=pl.coalesce("par_val", "m_val")).filter(pl.col("is_val") & ~pl.col("own_hidden").fill_null(False))
v = v.join(unc("val", rec), on="m", how="left").with_columns(scored=pl.col("pmax").is_not_null(), agree=pl.col("u_s1") == pl.col("s1"))
v = v.filter(~pl.col("tie_addr") & ~pl.col("tie_name"))
for ms in (0, 1):
    for thr_acc in (0.8, 0.9):
        res = v.filter((pl.col("mode_street") == ms) & ((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null())))
        out = [f"val guarded residual mode={'street' if ms else 'big'} pmax<{thr_acc}: groups {res.height}"]
        for t in (0.85, 0.9, 0.95):
            z = res.filter(pl.col("p_bin") >= t)
            zs, za = z.filter(pl.col("scored")), z.filter(pl.col("scored") & pl.col("agree").fill_null(False))
            out.append(f"p>={t}: all {z['label'].sum()}/{z.height} scored {zs['label'].sum()}/{zs.height} agree {za['label'].sum()}/{za.height}")
        print(" | ".join(out))
# OOF-wide (not residual) precision of scored/agree picks, by mode, p>=0.9
for ms in (0, 1):
    z = v.filter((pl.col("mode_street") == ms) & (pl.col("p_bin") >= 0.9) & pl.col("scored"))
    print(f"val all scored mode={ms} p>=0.9: {z['label'].sum()}/{z.height}; scored & pmax<0.9: {z.filter(pl.col('pmax') < 0.9)['label'].sum()}/{z.filter(pl.col('pmax') < 0.9).height}")

# ---- test
rect = ridmap("test")
ut = unc("test", rect)
tsc = pl.read_parquet(f"{W}/test_big_picks.parquet")
tsc = tsc.join(ut, on="m", how="left").with_columns(scored=pl.col("pmax").is_not_null(), agree=(pl.col("u_s1") == pl.col("s1")).fill_null(False))
tg = tsc.filter(~pl.col("tie_addr") & ~pl.col("tie_name"))
print("\nTEST guarded picks by mode, p band, scored/agree:")
print(tg.filter(pl.col("p_bin") >= 0.85).with_columns(b=pl.col("p_bin").cut([0.9, 0.95], left_closed=True)).group_by("country", "mode_street", "b", "scored", "agree").len()
      .sort("country", "mode_street", "b", "scored", "agree"))

# ---- final selection: guarded, production-scored record whose production best S1 == our pick, p >= 0.9; caps vs fr30
sel = tg.filter((pl.col("p_bin") >= 0.9) & pl.col("scored") & pl.col("agree") & (pl.col("country") != "France")).with_columns(
    kind=pl.format("{}_{}", pl.when(pl.col("mode_street") == 1).then(pl.lit("street")).otherwise(pl.lit("big")), pl.col("kind")))
print("chosen pmax (production best prob) quantiles:", sel["pmax"].quantile(0.1), sel["pmax"].median(), sel["pmax"].max())
sel = sel.select("m", "s1", "country", "kind", pl.col("p_bin").alias("p"), "src")
g = pl.read_csv("output/v28q_fr30/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
ex = g.select(s1=pl.col("source1_entity_id"), m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
assert sel.join(ex, on="m").height == 0, "already assigned"
assert sel["m"].n_unique() == sel.height
cnt = ex.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("s1", "src").len()
sel = sel.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0)).sort("p", descending=True)
sel = sel.with_columns(k=pl.int_range(pl.len()).over("s1", "src") + 1)
sel = sel.with_columns(ok=(pl.col("len") + pl.col("k")) <= pl.when(pl.col("src") == "S2").then(5).otherwise(6))
print("cap drops:", sel.filter(~pl.col("ok")).height)
sel = sel.filter(pl.col("ok")).select("m", "s1", "country", "kind", "p")
print(sel.group_by("country", "kind").agg(pl.len(), pl.col("p").min().round(3)).sort("country", "kind"))
if "--write" in sys.argv:
    sel.write_parquet("local_data/work/agent_mt_adds_big.parquet")
    print("wrote", sel.height)
