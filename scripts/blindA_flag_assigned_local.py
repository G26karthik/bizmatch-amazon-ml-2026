"""blindA (local, test): record-level noise flags -> share of raw records assigned by the submission, per country.
Also train: matched share per flag (truth) for US/India reference."""
import polars as pl, sys, gc
R = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
def rd(split, k):
    return pl.read_csv(f"{R}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
def fl(df):
    n, a = pl.col("business_name"), pl.col("business_address")
    return df.select("entity_id", "country",
        (a == "").alias("a_empty"), ((a != "") & ~a.str.contains(r"\d")).alias("a_nodigit"),
        ((n == n.str.to_lowercase()) & n.str.contains(r"[a-z]")).alias("n_lower"),
        ((n == n.str.to_uppercase()) & n.str.contains(r"[A-Z]")).alias("n_upper"),
        n.str.contains(r"(?i)\.com|www\.").alias("n_web"), n.str.contains("  ").alias("n_dbl"),
        n.str.contains(r"^[^\w\s]").alias("n_lead_punct"), n.str.contains(r"[\[]").alias("n_sqbracket"),
        (~n.str.contains(" ")).alias("n_one_token"), n.str.contains(r"\d").alias("n_digit"),
        pl.lit(True).alias("all"))
out = []
for split, lab in [("train", "truth"), ("test", "sub")]:
    recs = pl.concat([fl(rd(split, 2)), fl(rd(split, 3))])
    if split == "train":
        gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        m = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("entity_id")).filter(pl.col("entity_id") != "")
    else:
        sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
        m = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("entity_id")).filter(pl.col("entity_id") != "")
    recs = recs.join(m.with_columns(pl.lit(True).alias("asg")), on="entity_id", how="left").with_columns(pl.col("asg").fill_null(False))
    ns1 = rd(split, 1).group_by("country").len().rename({"len": "ns1"})
    flags = [c for c in recs.columns if c not in ("entity_id", "country", "asg")]
    for f in flags:
        g = recs.filter(pl.col(f)).group_by("country").agg(pl.len().alias("n"), pl.col("asg").mean().alias("asg_frac")).join(ns1, on="country")
        g = g.with_columns((pl.col("n") / pl.col("ns1")).alias("rec_per_s1"), (pl.col("n") * pl.col("asg_frac") / pl.col("ns1")).alias("asg_per_s1"), pl.lit(f).alias("flag"), pl.lit(f"{split}_{lab}").alias("set"))
        out.append(g.select("set", "country", "flag", "rec_per_s1", "asg_per_s1", "asg_frac"))
    del recs; gc.collect()
T = pl.concat(out).with_columns((pl.col("set") + "_" + pl.col("country")).alias("key"))
for v in ["rec_per_s1", "asg_frac"]:
    P = T.pivot(on="key", index="flag", values=v).with_columns(pl.exclude("flag").round(4))
    print(f"\n### {v}"); print(P)
