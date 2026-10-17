"""blindA (local, train): copy-of-copy chains? Do two copies of the same S1 share novel (non-S1) name tokens or
noise flags more than independent noise would give?"""
import polars as pl, sys
R = sys.argv[1]
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
def rd(f):
    return pl.read_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "business_name", "business_address"]).fill_null("")
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
del gt
s1 = rd("train_source1.tsv").select(pl.col("entity_id").alias("s1"), norm("business_name").alias("sn"))
m = pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("raw"), norm("business_name").alias("mn"), (pl.col("business_address") == "").alias("aemp"))
p = pairs.join(s1, on="s1").join(m, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
p = p.with_columns(pl.col("mn").str.split(" ").list.set_difference(pl.col("sn").str.split(" ")).alias("nov"),
                   (pl.col("raw") == pl.col("raw").str.to_lowercase()).alias("lower"),
                   pl.col("raw").str.contains(r"\.com|www\.").alias("web"),
                   pl.col("raw").str.contains(r"[\(\[]").alias("brk"),
                   pl.col("raw").str.contains("  ").alias("dbl"))
p = p.with_columns(pl.int_range(pl.len()).over(["s1"]).alias("i"))
a = p.filter(pl.col("i") == 0); b = p.filter(pl.col("i") == 1)
x = a.join(b, on="s1", suffix="_b")
# control: pair copy i=0 of S1 with copy i=1 of a different random S1 (shifted)
bb = b.with_columns(pl.col("s1").shift(1).alias("s1x")).drop_nulls("s1x")
ctl = a.join(bb.select(pl.col("s1x").alias("s1"), pl.col("nov").alias("nov_b"), pl.col("lower").alias("lower_b"), pl.col("web").alias("web_b"), pl.col("brk").alias("brk_b"), pl.col("dbl").alias("dbl_b"), pl.col("aemp").alias("aemp_b"), pl.col("raw").alias("raw_b")), on="s1")
for nm, d in [("same S1 (copy0, copy1)", x), ("control (different S1)", ctl)]:
    d = d.with_columns(pl.col("nov").list.set_intersection(pl.col("nov_b")).list.len().alias("shared_nov"))
    out = {"n": d.height,
           "P(both have novel tok)": round(((pl.col("nov").list.len() > 0) & (pl.col("nov_b").list.len() > 0)).mean().item() if False else float(((d["nov"].list.len() > 0) & (d["nov_b"].list.len() > 0)).mean()), 4),
           "P(share a novel tok)": round(float((d["shared_nov"] > 0).mean()), 5),
           "P(raw names identical)": round(float((d["raw"] == d["raw_b"]).mean()), 5)}
    for f in ["lower", "web", "brk", "dbl", "aemp"]:
        pa, pb, pj = float(d[f].mean()), float(d[f + "_b"].mean()), float((d[f] & d[f + "_b"]).mean())
        out[f] = f"{pj:.5f} vs indep {pa*pb:.5f}"
    print(nm, out)
# which novel tokens are shared most for same-S1 pairs (legal words etc. expected)
y = x.with_columns(pl.col("nov").list.set_intersection(pl.col("nov_b")).alias("sh")).select("sh").explode("sh").drop_nulls().group_by("sh").len().sort("len", descending=True)
print("most shared novel tokens:", y.head(25).to_dicts())
