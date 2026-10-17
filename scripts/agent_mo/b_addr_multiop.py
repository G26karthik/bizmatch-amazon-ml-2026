"""agent_mo task B: ADDRESSED records (US/India) whose address is format-tolerant equal to EXACTLY ONE S1 of the country
(house-number tokens equal after zero-stripping, non-empty; full-token Jaccard >= 0.35; blocking on house + a street token
that is non-digit, len>=3, in <=1% of the country's S1 addresses), and whose name (tokens minus legal words) shares >= 2
RARE tokens (frequency among the country's S1 names < 200) with that S1 but differs by >= 2 token operations
(ops = max(|R\\S|, |S\\R|)).
Train (all visible, 50% record sample) precision by class; test all records with fr27 status; decoy-excess check.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_mo/b_addr_multiop.py
"""
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
LEG = ["inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"]
OUT = "local_data/work/agent_mo_B_{}.parquet"
TR_FRAC = 0.5
CHUNK = 150_000


def tok(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
            .str.extract_all(r"\w+").list.unique())


def prep(df):
    df = df.with_columns(tok("business_address").alias("at"), tok("business_name").alias("nt"))
    return df.with_columns(
        pl.col("at").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.replace(r"^0+", "")).list.sort().list.join(" ").alias("house"),
        pl.col("nt").list.set_difference(pl.lit(LEG)).alias("nk"))


def run(split, ctry, frac=1.0):
    rec = pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet").select("entity_id", "business_name", "business_address", "country")
                     .filter(pl.col("country") == ctry).drop("country") for s in (1, 2, 3)]).collect()
    s1 = prep(rec.filter(pl.col("entity_id").str.starts_with("S1")))
    oth = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
    del rec
    if frac < 1:
        oth = oth.sample(fraction=frac, seed=11)
    nS1 = s1.height
    adf = s1.select(pl.col("at").alias("t")).explode("t").group_by("t").len()
    common = adf.filter(pl.col("len") > 0.01 * nS1)["t"]
    ndf = s1.select(pl.col("nk").alias("t")).explode("t").group_by("t").len().rename({"len": "nfreq"})

    def blocks(df, idc):
        return (df.filter(pl.col("house") != "").select(idc, "house", pl.col("at").alias("t")).explode("t")
                .filter(~pl.col("t").str.contains(r"\d") & (pl.col("t").str.len_chars() >= 3) & ~pl.col("t").is_in(common.implode()) & ~pl.col("t").is_in(["bis", "ter"])))
    sb = blocks(s1, "entity_id").rename({"entity_id": "s1"})
    s1s = s1.select(pl.col("entity_id").alias("s1"), pl.col("at").alias("at1"), pl.col("nk").alias("nk1"), pl.col("business_name").alias("name1"), pl.col("business_address").alias("addr1"))
    outs = []
    for i in range(0, oth.height, CHUNK):
        q = prep(oth.slice(i, CHUNK))
        qb = blocks(q, "entity_id").rename({"entity_id": "m"})
        pr = qb.join(sb, on=["house", "t"]).select("m", "s1").unique()
        pr = pr.join(q.select(pl.col("entity_id").alias("m"), "at"), on="m").join(s1s.select("s1", "at1"), on="s1")
        pr = pr.with_columns((pl.col("at").list.set_intersection("at1").list.len() / pl.col("at").list.set_union("at1").list.len()).alias("J")).filter(pl.col("J") >= 0.35)
        pr = pr.select("m", "s1", "J").filter(pl.len().over("m") == 1)
        pr = pr.join(q.select(pl.col("entity_id").alias("m"), "at", "nk", "business_name", "business_address"), on="m").join(s1s, on="s1")
        pr = pr.with_columns(pl.col("nk").list.set_difference("nk1").list.len().alias("a"), pl.col("nk1").list.set_difference("nk").list.len().alias("d"),
                             pl.col("nk").list.set_intersection("nk1").alias("sh"))
        pr = pr.with_columns(pl.max_horizontal("a", "d").alias("ops")).filter(pl.col("ops") >= 2)
        rs = pr.select("m", pl.col("sh").alias("t")).explode("t").join(ndf, on="t", how="left").group_by("m").agg(
            (pl.col("nfreq").fill_null(0) < 200).sum().alias("nrare"), pl.len().alias("nsh"))
        pr = pr.join(rs, on="m", how="left").filter(pl.col("nrare") >= 2)
        outs.append(pr.select("m", "s1", "ops", "a", "d", "nsh", "nrare", "J", pl.col("nk").list.len().alias("nr"), pl.col("nk1").list.len().alias("ns"),
                              "business_name", "name1", "business_address", "addr1"))
        print(split, ctry, i, pr.height, flush=True)
    return pl.concat(outs).with_columns(pl.lit(ctry).alias("country")), nS1


import sys
split, c = sys.argv[1], sys.argv[2]
t, n = run(split, c, TR_FRAC if split == "train" else 1.0)
if split == "train":
    tp = pl.scan_parquet("local_data/train_pairs.parquet").filter(pl.col("m").is_in(t["m"].implode())).collect()
    t = t.join(tp.rename({"s1": "s1_true"}), on="m", how="left").with_columns((pl.col("s1_true") == pl.col("s1")).fill_null(False).alias("right"))
else:
    g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_fr27", "matched_entity_ids": "m"})
    t = t.join(acc, on="m", how="left").with_columns(
        pl.when(pl.col("s1_fr27").is_null()).then(pl.lit("unassigned")).when(pl.col("s1_fr27") == pl.col("s1")).then(pl.lit("acc_same")).otherwise(pl.lit("acc_other")).alias("status"))
t.write_parquet(OUT.format(f"{split}_{c}"))
print("done", split, c, n, t.height, flush=True)
