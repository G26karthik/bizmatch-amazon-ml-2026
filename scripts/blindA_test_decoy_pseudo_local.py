import polars as pl, sys
C = sys.argv[1]
toks = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
for split in ["train", "test"]:
    d = pl.read_parquet(f"{C}/{split}_US_decrec.parquet").with_columns(toks("name").alias("rt"), toks("s1_name").alias("st"))
    d = d.with_columns((pl.col("rt").list.len() == 1).alias("one_tok"), (pl.col("rt").list.set_intersection(pl.col("st")).list.len() == 0).alias("no_overlap"))
    print(split, "US decoy-offset records (name adds a word):", d.height, "one-token:", round(float(d["one_tok"].mean()), 5), "one-token & no overlap:", round(float((d["one_tok"] & d["no_overlap"]).mean()), 5))
    print(d.filter(pl.col("one_tok")).select("name", "s1_name", "d").head(8))
