"""Does every copy carry a visible name operation? On train truth (US/India): share of true copies whose raw name is
byte-identical to the S1's raw name, identical ignoring case, identical ignoring case+spaces+punctuation+accents.
Then the same shares for France test accepted pairs by house relation (fr14): same house vs moved house.

uv run --no-project --with polars==1.44.2 python scripts/raw_ident.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)


def norm(s):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower())


def rel(a, b):
    if a == b:
        return "0_raw_identical"
    if a.lower() == b.lower():
        return "1_case_only"
    if " ".join(a.lower().split()) == " ".join(b.lower().split()):
        return "2_space_case"
    if norm(a) == norm(b):
        return "3_punct_accent"
    return "4_other"


s1 = pl.read_parquet("local_data/train_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                                                 pl.col("business_address").alias("a1"), "country")
oth = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).alias("src")) for s in (2, 3)], how="vertical_relaxed") \
        .select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"), "src")
gt = pl.read_parquet("local_data/train_pairs.parquet").join(s1, on="s1").join(oth, on="m")
gt = gt.with_columns(pl.Series("rel", [rel(a, b) for a, b in zip(gt["n1"].to_list(), gt["n2"].to_list())]),
                     (pl.col("a2").is_null() | (pl.col("a2").str.strip_chars() == "")).alias("empty"))
print("=== train truth: name relation of TRUE copies to their S1")
print(gt.group_by("country", "rel").agg(pl.len()).with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share")).sort("country", "rel"))
print(gt.group_by("country", "empty", "rel").agg(pl.len()).with_columns((pl.col("len") / pl.col("len").sum().over(["country", "empty"])).round(4).alias("share"))
        .filter(pl.col("rel") == "0_raw_identical").sort("country", "empty"))

# test France accepted pairs
t1 = pl.read_parquet("local_data/test_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                                                pl.col("business_address").alias("a1"), "country")
t2 = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet").with_columns(pl.lit(s).alias("src")) for s in (2, 3)], how="vertical_relaxed") \
       .select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"), "src")
g = pl.read_csv("output/v28q_fr14/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}).join(t1, on="s1").join(t2, on="m")


def house(a):
    m = re.search(r"\b(\d{1,5})\b", a or "")
    return int(m.group(1)) if m else None


def street(a, h):
    t = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", a or "").encode("ascii", "ignore").decode().lower())
    return " ".join(sorted(set(t)))


OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
hr = []
for a1, a2 in zip(acc["a1"].to_list(), acc["a2"].to_list()):
    if not a2 or not a2.strip():
        hr.append("no_addr"); continue
    h1, h2 = house(a1), house(a2)
    if h1 is None or h2 is None:
        hr.append("no_house"); continue
    d = h2 - h1
    hr.append("same_house" if d == 0 else ("offset" if d in OFFS else ("negative" if d < 0 else "pos_other")))
acc = acc.with_columns(pl.Series("hrel", hr), pl.Series("rel", [rel(a, b) for a, b in zip(acc["n1"].to_list(), acc["n2"].to_list())]))
print("\n=== test accepted pairs (fr14): name relation by house relation")
for c in ("France", "US", "India"):
    z = acc.filter(pl.col("country") == c)
    print(f"-- {c} ({z.height} pairs)")
    print(z.group_by("hrel", "rel").agg(pl.len()).with_columns((pl.col("len") / pl.col("len").sum().over("hrel")).round(4).alias("share"))
          .filter(pl.col("rel").is_in(["0_raw_identical", "1_case_only"])).sort("hrel", "rel"))
acc.filter(pl.col("country") == "France").select("s1", "m", "src", "hrel", "rel", "n1", "n2", "a1", "a2").write_parquet(
    "local_data/work/fr_acc_rel.parquet")
