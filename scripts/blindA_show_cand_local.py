import polars as pl, sys
C = sys.argv[1]; R = sys.argv[2]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(58)
c = pl.read_csv(f"{C}/blindA_fr_pseudo_initials_candidates.tsv", separator="\t", infer_schema_length=0)
r = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "addr")
s = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("addr").alias("s1_addr"))
x = c.join(r, on="entity_id").join(s, on="s1")
print(x.group_by("cls").len())
print(x.sample(30, seed=7).select("cls", "name", "s1_name", "addr", "s1_addr", "k_pred_other"))
