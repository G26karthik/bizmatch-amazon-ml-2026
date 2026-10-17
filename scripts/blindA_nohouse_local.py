"""blindA (local): records with a non-empty address but no house number (decoys always carry a moved house number).
Candidates = S1s on the same street key; class by name evidence (best vs second-best token coverage).
US train truth precision per class vs France test assignment per class."""
import polars as pl, sys
C = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
LEG = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|private|llp|lp|pc|pllc|plc|sarl|sas|sasu|sa|eurl|sci|snc|ei)\b"
def nt(c):
    return pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(LEG, " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
def classify(j):
    j = j.with_columns(nt("name").alias("rt"), nt("s1_name").alias("st"))
    j = j.with_columns((pl.col("rt").list.set_intersection(pl.col("st")).list.len() / pl.col("st").list.len().clip(1, None)).alias("cov"),
                       (pl.col("rt").list.sort() == pl.col("st").list.sort()).alias("eq"))
    j = j.sort(["entity_id", "cov"], descending=[False, True])
    g = j.group_by("entity_id", maintain_order=True).agg(pl.col("s1").first().alias("best"), pl.col("cov").first().alias("c1"), pl.col("eq").first().alias("eq1"),
                                                          pl.col("cov").get(1, null_on_oob=True).alias("c2") if hasattr(pl.Expr, "get") else pl.col("cov").slice(1, 1).first().alias("c2"), pl.len().alias("ncand"))
    g = g.with_columns(pl.col("c2").fill_null(0.0))
    g = g.with_columns(pl.when(pl.col("eq1") & (pl.col("c2") < 1.0)).then(pl.lit("A_exact_unique"))
                       .when((pl.col("c1") >= 0.99) & (pl.col("c2") < 0.5)).then(pl.lit("B_cov1_clear"))
                       .when((pl.col("c1") >= 0.5) & (pl.col("c2") < 0.5)).then(pl.lit("C_cov50_clear"))
                       .when(pl.col("c1") >= 0.5).then(pl.lit("D_ambiguous"))
                       .otherwise(pl.lit("E_weak")).alias("cls"))
    return g
# US train truth (crude US key)
recs = pl.read_parquet(f"{C}/us_recs.parquet").filter(pl.col("h").is_null() & (pl.col("addr") != "") & (pl.col("K") != ""))
s1 = pl.read_parquet(f"{C}/us_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K")
g = classify(recs.select("entity_id", "name", "K").join(s1, on="K")).join(recs.select("entity_id", "true_s1"), on="entity_id")
g = g.with_columns((pl.col("best") == pl.col("true_s1")).fill_null(False).alias("best_true"), pl.col("true_s1").is_not_null().alias("any_true"))
print("US TRAIN no-house records with a same-street S1:", g.height)
print(g.group_by("cls").agg(pl.len(), pl.col("best_true").mean().round(4), pl.col("any_true").mean().round(4)).sort("cls"))
# France test
fr = pl.read_parquet(f"{C}/test_FR_recs.parquet").filter(pl.col("h").is_null() & (pl.col("addr") != "") & (pl.col("K") != ""))
fs1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K")
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref", "matched_entity_ids": "entity_id"})
gf = classify(fr.select("entity_id", "name", "K").join(fs1, on="K")).join(sp, on="entity_id", how="left")
gf = gf.with_columns((pl.col("ref") == pl.col("best")).fill_null(False).alias("to_best"), pl.col("ref").is_not_null().alias("any"))
print("\nFRANCE TEST no-house records with a same-street S1:", gf.height, "(no-house France records total:", fr.height, ")")
print(gf.group_by("cls").agg(pl.len(), pl.col("to_best").mean().round(4), pl.col("any").mean().round(4)).sort("cls"))
un = gf.filter(~pl.col("any") & pl.col("cls").is_in(["A_exact_unique", "B_cov1_clear"]))
print("France unassigned no-house records in classes A/B:", un.height)
print(un.join(fr.select("entity_id", "name", "addr"), on="entity_id").join(fs1.select(pl.col("s1").alias("best"), "s1_name"), on="best").select("name", "s1_name", "addr", "c1", "c2", "ncand").head(25))
un.write_parquet(f"{C}/fr_nohouse_unassigned_AB.parquet")
