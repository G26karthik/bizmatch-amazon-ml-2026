"""blindA (local): noise-operation mix of matched pairs per S1: train truth (US, India) vs test submission (US, India, France).
If the generator's op mix is country-agnostic, test/train per-S1 ratios localise recall deficits by op class."""
import polars as pl, sys, gc
R = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
def rd(split, k):
    return pl.read_csv(f"{R}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
def flags(pairs, split):
    s1 = rd(split, 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa"), "country")
    m = pl.concat([rd(split, 2), rd(split, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("mn"), pl.col("business_address").alias("ma"))
    p = pairs.join(s1, on="s1").join(m, on="m")
    del m; gc.collect()
    snn, mnn = norm("sn"), norm("mn")
    p = p.with_columns(snn.alias("snn"), mnn.alias("mnn"))
    st = pl.col("snn").str.split(" "); mt = pl.col("mnn").str.split(" ")
    p = p.select("s1", "country", pl.col("m").str.slice(0, 2).alias("src"),
        (pl.col("ma") == "").alias("a_empty"),
        ((pl.col("ma") != "") & ~pl.col("ma").str.contains(r"\d")).alias("a_nodigit"),
        (pl.col("snn") == pl.col("mnn")).alias("n_ident"),
        ((pl.col("mn") == pl.col("mn").str.to_lowercase()) & pl.col("mn").str.contains(r"[a-z]")).alias("n_lower"),
        ((pl.col("mn") == pl.col("mn").str.to_uppercase()) & pl.col("mn").str.contains(r"[A-Z]")).alias("n_upper"),
        pl.col("mn").str.contains(r"(?i)\.com|www\.").alias("n_web"),
        (pl.col("mn").str.contains(r"[\(\[]") & ~pl.col("sn").str.contains(r"[\(\[]")).alias("n_bracket_new"),
        pl.col("mn").str.contains("  ").alias("n_dblspace"),
        pl.col("mn").str.contains(r"^[^\w\s]").alias("n_lead_punct"),
        (mt.list.len() < st.list.len()).alias("n_fewer_tok"),
        (mt.list.len() > st.list.len()).alias("n_more_tok"),
        ((mt.list.len() == st.list.len()) & (pl.col("snn") != pl.col("mnn"))).alias("n_same_len_changed"),
        (mt.list.set_intersection(st).list.len() == 0).alias("n_no_shared_tok"),
        (pl.col("mn").str.contains(r"[^\x00-\x7f]") & ~pl.col("sn").str.contains(r"[^\x00-\x7f]")).alias("n_nonascii_new"),
        (pl.col("mn").str.contains(r"\d") & ~pl.col("sn").str.contains(r"\d")).alias("n_digit_new"),
        (pl.col("ma") == pl.col("sa")).alias("a_exact"),
    )
    return p
out = {}
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
tp = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
ns1 = gt.join(rd("train", 1).select(pl.col("entity_id").alias("source1_entity_id"), "country"), on="source1_entity_id").group_by("country").len()
del gt; gc.collect()
f = flags(tp, "train"); del tp; gc.collect()
cols = [c for c in f.columns if c not in ("s1", "country", "src")]
agg = f.group_by("country").agg([pl.len().alias("pairs")] + [pl.col(c).sum() for c in cols]).join(ns1, on="country")
agg = agg.with_columns(pl.lit("train").alias("split"))
del f; gc.collect()
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
ns1t = rd("test", 1).group_by("country").len()
g = flags(sp, "test")
agg2 = g.group_by("country").agg([pl.len().alias("pairs")] + [pl.col(c).sum() for c in cols]).join(ns1t, on="country").with_columns(pl.lit("test_sub").alias("split"))
A = pl.concat([agg, agg2])
A = A.with_columns([(pl.col(c) / pl.col("len")).alias(c) for c in ["pairs"] + cols])
T = A.with_columns((pl.col("split") + "_" + pl.col("country")).alias("key")).drop(["split", "country", "len"]).unpivot(index="key", variable_name="flag", value_name="per_s1").pivot(on="key", index="flag", values="per_s1")
T = T.with_columns((pl.col("test_sub_US") / pl.col("train_US")).round(3).alias("US_test/train"),
                   (pl.col("test_sub_India") / pl.col("train_India")).round(3).alias("IN_test/train"),
                   (pl.col("test_sub_France") / pl.col("train_US")).round(3).alias("FR/trainUS"),
                   (pl.col("test_sub_France") / pl.col("test_sub_US")).round(3).alias("FR/testUS"))
print(T.with_columns([pl.col(c).round(4) for c in ["train_US", "train_India", "test_sub_US", "test_sub_India", "test_sub_France"]]))
