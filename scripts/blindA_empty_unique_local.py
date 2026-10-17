"""blindA (local): empty-address S2/S3 records.
Train truth: P(copy of the unique exact-name S1) etc.  Test: how many such records the submission leaves unassigned,
with cap / emptiness context.  Normalisation variants: plain (lower, alnum) and legal-stripped."""
import polars as pl, sys, gc
R = sys.argv[1]; SUB = sys.argv[2]; OUT = sys.argv[3]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
LEGAL = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|private|llp|lp|pc|pllc|plc|sarl|sas|sasu|sa|eurl|sci|snc|ei)\b"
def nm(c, legal=False):
    e = pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ")
    if legal:
        e = e.str.replace_all(LEGAL, " ")
    return e.str.replace_all(r"\s+", " ").str.strip_chars()
def rd(split, k):
    return pl.read_csv(f"{R}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
for split in ["train", "test"]:
    s1 = rd(split, 1).select(pl.col("entity_id").alias("s1"), "country", nm("business_name").alias("n0"), nm("business_name", True).alias("n1"), pl.col("business_name").alias("s1_name"))
    emp = pl.concat([rd(split, k).filter(pl.col("business_address") == "") for k in [2, 3]]).select("entity_id", "country", pl.col("business_name").alias("name"), nm("business_name").alias("n0"), nm("business_name", True).alias("n1"))
    if split == "train":
        gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        pr = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref_s1", "matched_entity_ids": "entity_id"})
    else:
        sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
        pr = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref_s1", "matched_entity_ids": "entity_id"})
    pr = pr.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    cnt = pr.group_by(["ref_s1", "src"]).len().pivot(on="src", index="ref_s1", values="len").fill_null(0)
    emp = emp.join(pr.select("entity_id", "ref_s1"), on="entity_id", how="left").with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    for key in ["n0", "n1"]:
        g = s1.group_by(["country", key]).agg(pl.len().alias("ns"), pl.col("s1").first().alias("cand"))
        e = emp.join(g, on=["country", key], how="left").with_columns(pl.col("ns").fill_null(0))
        e = e.with_columns(pl.when(pl.col("ns") == 1).then(pl.col("cand")).otherwise(None).alias("cand"))
        e = e.with_columns((pl.col("ref_s1") == pl.col("cand")).fill_null(False).alias("ref_is_cand"))
        lab = "true" if split == "train" else "sub"
        print(f"\n[{split} {key}] empty-address records by exact-name S1 count:")
        print(e.group_by(["country", pl.col("ns").clip(0, 3)]).agg(pl.len(), pl.col("ref_s1").is_not_null().mean().round(4).alias(f"{lab}_any"),
              pl.col("ref_is_cand").mean().round(4).alias(f"{lab}_to_unique_cand")).sort(["country", "ns"]))
        if split == "test" and key == "n0":
            u = e.filter((pl.col("ns") == 1) & pl.col("ref_s1").is_null())
            u = u.join(cnt.rename({"ref_s1": "cand"}), on="cand", how="left").with_columns(pl.col("S2").fill_null(0), pl.col("S3").fill_null(0))
            u = u.with_columns(((pl.col("src") == "S2") & (pl.col("S2") >= 5) | (pl.col("src") == "S3") & (pl.col("S3") >= 6)).alias("cap_full"),
                               ((pl.col("S2") + pl.col("S3")) == 0).alias("cand_empty"))
            print(" unassigned unique-exact-name empty-address records:", u.group_by("country").agg(pl.len(), pl.col("cap_full").sum(), pl.col("cand_empty").sum()).to_dicts())
            u = u.join(s1.select(pl.col("s1").alias("cand"), "s1_name"), on="cand")
            print(u.select("country", "name", "s1_name", "S2", "S3").head(25))
            u.write_parquet(OUT)
        if split == "train" and key == "n0":
            # cap context in train truth: for ns==1 records, is the unique cand the true S1 depending on cand's other copies
            u = e.filter(pl.col("ns") == 1).join(cnt.rename({"ref_s1": "cand"}), on="cand", how="left").with_columns(pl.col("S2").fill_null(0), pl.col("S3").fill_null(0))
            u = u.with_columns(((pl.col("S2") + pl.col("S3")) - pl.col("ref_is_cand").cast(pl.Int32)).alias("other_copies"))
            print(" train ns==1: P(true=cand) by cand's other true copies:", u.group_by(pl.col("other_copies").clip(0, 6)).agg(pl.len(), pl.col("ref_is_cand").mean().round(4)).sort("other_copies").to_dicts())
            print(" train ns==1 wrong cases: matched elsewhere", (u["ref_s1"].is_not_null() & ~u["ref_is_cand"]).sum(), "unmatched", u["ref_s1"].is_null().sum())
    del s1, emp, pr; gc.collect()
