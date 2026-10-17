"""agent_mt: street-mode production-residual check at high p bands, split by n_cand and record house-number presence."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
W = "local_data/work/agent_mt"
pk = pl.read_parquet(f"{W}/oof_big_picks.parquet")
SEED, VAL_MOD, HIDE = 42, 10, 19
rec = pl.concat([pl.scan_parquet(f"local_data/train_source{s}.parquet").select("entity_id").with_columns(pl.lit(s).cast(pl.Int8).alias("src_n")) for s in (1, 2, 3)]).with_row_index("rid").collect()
rec = rec.with_columns(hidden=(pl.col("src_n") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE), own_val=pl.col("entity_id").hash(SEED) % VAL_MOD == 0)
tp = pl.read_parquet("local_data/train_pairs.parquet").join(pk.select("m"), on="m")
s1i = rec.filter(pl.col("src_n") == 1).select(pl.col("entity_id").alias("owner"), pl.col("hidden").alias("own_hidden"), pl.col("own_val").alias("par_val"))
v = pk.join(tp.rename({"s1": "owner"}), on="m", how="left").join(s1i, on="owner", how="left")
v = v.join(rec.select(pl.col("entity_id").alias("m"), pl.col("rid").alias("q"), pl.col("own_val").alias("m_val")), on="m", how="left")
v = v.with_columns(is_val=pl.coalesce("par_val", "m_val")).filter(pl.col("is_val") & ~pl.col("own_hidden").fill_null(False))
uv = pl.read_parquet("local_data/llm/unc_val.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("prob").cast(pl.Float64)).group_by("q").agg(pmax=pl.col("prob").max())
v = v.join(uv, on="q", how="left").filter(pl.col("mode_street") == 1 & ~pl.col("tie_addr") & ~pl.col("tie_name"))
for thr_acc in (0.8, 0.9):
    res = v.filter((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null()))
    print(f"\nstreet guarded prod-residual pmax<{thr_acc}: groups {res.height}")
    for t in (0.9, 0.95, 0.98, 0.99, 0.995):
        z = res.filter(pl.col("p_bin") >= t)
        print(f"  p>={t}: {z['label'].sum()}/{z.height} noise={(z['m_has_owner'] == 0).sum()}  | n_cand=1: {z.filter(pl.col('n_cand') == 1)['label'].sum()}/{z.filter(pl.col('n_cand') == 1).height}"
              f"  n_cand>1: {z.filter(pl.col('n_cand') > 1)['label'].sum()}/{z.filter(pl.col('n_cand') > 1).height}  rec_has_h: {z.filter(pl.col('rec_has_h') == 1)['label'].sum()}/{z.filter(pl.col('rec_has_h') == 1).height}")
    print(res.filter(pl.col("p_bin") >= 0.9).select("mname", "sname", "p_bin", "label", "m_has_owner", "n_cand", "rec_has_h", "cand_has_h", "pmax", "kind").sort("p_bin"))
