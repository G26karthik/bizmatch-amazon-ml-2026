"""agent_mt: test-like residual check using production uncertain-zone validation probabilities (local_data/llm/unc_val.parquet,
same split as scripts/deep/testlike_rule.py: SEED 42, VAL_MOD 10, 19% hidden). Residual = val records production would NOT
accept (uncertain-zone best prob < THR_ACC, or never scored and not certain). Hidden-owner records dropped (test has none)."""
import sys
import polars as pl
sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks  # noqa: E402
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
SEED, VAL_MOD, HIDE = 42, 10, 19
rec = pl.concat([pl.scan_parquet(f"local_data/train_source{s}.parquet").select("entity_id").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid").collect()
rec = rec.with_columns(hidden=(pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE),
                       own_val=pl.col("entity_id").hash(SEED) % VAL_MOD == 0)
pk = picks(pl.read_parquet(f"{W}/oof.parquet"), "p_bin")
tp = pl.read_parquet("local_data/train_pairs.parquet").join(pk.select("m"), on="m")
s1i = rec.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), pl.col("hidden").alias("own_hidden"), pl.col("own_val").alias("par_val"))
pk = pk.join(tp.rename({"s1": "owner"}), on="m", how="left").join(s1i.rename({"s1": "owner"}), on="owner", how="left")
pk = pk.join(rec.select(pl.col("entity_id").alias("m"), pl.col("rid").alias("q"), pl.col("own_val").alias("m_val")), on="m", how="left")
pk = pk.with_columns(is_val=pl.coalesce("par_val", "m_val"))
uv = pl.read_parquet("local_data/llm/unc_val.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("prob").cast(pl.Float64))
uvq = uv.group_by("q").agg(pmax=pl.col("prob").max())
print("uv records:", uvq.height)
pk = pk.join(uvq.with_columns(pl.col("q").cast(pl.UInt32)), on="q", how="left")
v = pk.filter(pl.col("is_val") & ~pl.col("own_hidden").fill_null(False))
print("val OOF groups (owner not hidden):", v.height, " in uv:", v["pmax"].is_not_null().sum())
for thr_acc in (0.5, 0.7):
    res = v.filter((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null()))
    print(f"\nresidual (prod pmax < {thr_acc} or never-scored noise): groups {res.height}, noise {(res['m_has_owner'] == 0).sum()}, owner-in-cands {res['has_pos'].mean():.3f}")
    for t in (0.8, 0.85, 0.9, 0.95, 0.98):
        z = res.filter(pl.col("p_bin") >= t)
        print(f"  p>={t}: n={z.height} prec={z['label'].mean() if z.height else float('nan'):.4f} noise={(z['m_has_owner'] == 0).sum()}")
acc = v.filter(pl.col("pmax").is_null() & pl.col("owner").is_not_null())
print("\ncertain (prod accepts, not in uv):", acc.height, "of", v.height)
for thr_acc in (0.8, 0.9):
    res = v.filter((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null()))
    print(f"\nresidual (prod pmax < {thr_acc} or never-scored noise): groups {res.height}, noise {(res['m_has_owner'] == 0).sum()}")
    for t in (0.8, 0.85, 0.9, 0.95):
        z = res.filter(pl.col("p_bin") >= t)
        print(f"  p>={t}: n={z.height} prec={z['label'].mean() if z.height else float('nan'):.4f} noise={(z['m_has_owner'] == 0).sum()}")
res = v.filter((pl.col("pmax") < 0.9) | (pl.col("pmax").is_null() & pl.col("owner").is_null()))
print("noise picks p quantiles:", res.filter(pl.col("m_has_owner") == 0)["p_bin"].quantile(0.5), res.filter(pl.col("m_has_owner") == 0)["p_bin"].max())
print(res.filter(pl.col("p_bin") >= 0.5).select("mname", "sname", "p_bin", "label", "pmax", "m_has_owner").sort("p_bin"))
tie = pl.read_parquet(f"{W}/oof_tie.parquet")
v2 = v.join(tie, on="m", how="left").filter(~pl.col("tie_addr"))
print("\n=== GUARDED (no address tie) ===")
for thr_acc in (0.7, 0.8, 0.9):
    res = v2.filter((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null()))
    print(f"residual (prod pmax < {thr_acc} or never-scored noise): groups {res.height}, noise {(res['m_has_owner'] == 0).sum()}")
    for t in (0.8, 0.85, 0.9, 0.95):
        z = res.filter(pl.col("p_bin") >= t)
        print(f"  p>={t}: n={z.height} right={z['label'].sum()} prec={z['label'].mean() if z.height else float('nan'):.4f}")
