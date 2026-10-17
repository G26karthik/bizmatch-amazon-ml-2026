import polars as pl, sys
C = sys.argv[1]
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
toks = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
j = pl.read_parquet(f"{C}/test_FR_join.parquet")
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet")
# exclude records that sit exactly at ANY France S1 address (they could be that S1's copies)
occ = s1.select("K", "h").unique().with_columns(pl.lit(True).alias("occupied"))
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "K", "h").join(occ, on=["K", "h"], how="left").with_columns(pl.col("occupied").fill_null(False))
j = j.join(recs.select("entity_id", "occupied"), on="entity_id", how="left")
j = j.with_columns(toks("name").alias("rt"), toks("s1_name").alias("st"))
j = j.with_columns(((pl.col("rt").list.len() == 1) & ~pl.col("name").str.contains(r"(?i)\.com|www\.|com$|^[@#]") & (pl.col("rt").list.set_intersection(pl.col("st")).list.len() == 0)).alias("pseudo"))
j = j.with_columns(pl.when(pl.col("d") == 0).then(pl.lit("same_house")).otherwise(pl.lit("offset")).alias("pos"))
print(j.group_by(["pos", "occupied"]).agg(pl.len(), pl.col("pseudo").mean().round(5), (pl.col("n_add") > 0).mean().round(4).alias("adds_word")).sort(["pos", "occupied"]))
print(j.filter((pl.col("pos") == "offset") & ~pl.col("occupied") & pl.col("pseudo")).select("name", "s1_name", "d").head(12))
