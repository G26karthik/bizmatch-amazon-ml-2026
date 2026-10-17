import polars as pl, sys
R = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70)
def rd(split, k):
    return pl.read_csv(f"{R}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
m = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("entity_id")).filter(pl.col("entity_id") != "")
recs = pl.concat([rd("test", 2), rd("test", 3)]).filter(~pl.col("business_name").str.contains(" "))
recs = recs.join(m.with_columns(pl.lit(True).alias("asg")), on="entity_id", how="left").with_columns(pl.col("asg").fill_null(False))
def cls(df):
    n = pl.col("business_name")
    return df.with_columns(pl.when(n.str.contains(r"(?i)\.com|www\.")).then(pl.lit("web_dot"))
        .when(n.str.contains(r"(?i)com$") & (n.str.len_chars() > 8)).then(pl.lit("ends_com"))
        .when(n.str.len_chars() <= 4).then(pl.lit("short<=4"))
        .when(n.str.contains(r"[A-Z].*[A-Z]") & (n == n.str.to_uppercase())).then(pl.lit("upper_word"))
        .otherwise(pl.lit("other")).alias("cls"))
recs = cls(recs)
print(recs.group_by(["country", "cls"]).agg(pl.len(), pl.col("asg").mean().round(3)).sort(["country", "cls"]))
for c in ["France", "US"]:
    print(c, "unassigned one-token examples:")
    print(recs.filter((pl.col("country") == c) & ~pl.col("asg")).sample(35, seed=1).select("business_name", "business_address", "cls"))
