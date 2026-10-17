"""Do decoys (unmatched records at decoy offsets / same house) ever carry one-token pseudo names or initials? US train truth."""
import polars as pl, sys
C = sys.argv[1]
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
toks = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
s1 = pl.read_parquet(f"{C}/us_s1.parquet")
kc = s1.group_by("K").len().rename({"len": "nK"})
j = pl.read_parquet(f"{C}/us_join.parquet").join(kc, on="K").filter(pl.col("nK") == 1)
j = j.with_columns(toks("name").alias("rt"), toks("s1_name").alias("st"))
j = j.with_columns((pl.col("rt").list.len() == 1).alias("one_tok"),
                   (pl.col("rt").list.set_intersection(pl.col("st")).list.len() == 0).alias("no_overlap"),
                   ((pl.col("rt").list.len() == 1) & (pl.col("rt").list.first() == pl.col("st").list.eval(pl.element().str.slice(0, 1)).list.join(""))).alias("initials"))
j = j.with_columns(pl.when((pl.col("true_s1") == pl.col("s1")) & (pl.col("d") == 0)).then(pl.lit("copy_samehouse"))
                   .when(pl.col("true_s1").is_null() & pl.col("d").is_in(OFF)).then(pl.lit("decoy_offset"))
                   .when(pl.col("true_s1").is_null() & (pl.col("d") == 0)).then(pl.lit("decoy_samehouse"))
                   .otherwise(pl.lit("other")).alias("cls"))
print(j.group_by("cls").agg(pl.len(), pl.col("one_tok").mean().round(5), (pl.col("one_tok") & pl.col("no_overlap")).mean().round(5).alias("pseudo_1tok"),
      pl.col("initials").mean().round(5), pl.col("no_overlap").mean().round(5)).sort("cls"))
print(j.filter((pl.col("cls") != "copy_samehouse") & pl.col("one_tok")).select("cls", "name", "s1_name", "d").head(20))
