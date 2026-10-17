"""blindA (local): empty-address true copies and their S1 namesakes: any tie-breaking artefact?
Tests: (a) copy-name tokens absent from the S1 name that occur in the S1 address;
       (b) raw-level differences among namesake S1s.
"""
import polars as pl, sys, re, collections
R = sys.argv[1]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(70)
def scan(f):
    return pl.scan_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False).rename({"business_name": "name", "business_address": "addr"}).with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null(""))
s1 = scan("train_source1.tsv").collect()
gt = pl.scan_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).collect().fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
del gt
emp = pl.concat([scan(f"train_source{k}.tsv").filter(pl.col("addr") == "").collect() for k in [2, 3]])
print("empty-address S2/S3 records:", emp.height)
emp = emp.join(pairs, left_on="entity_id", right_on="m", how="left")
print(" matched:", emp["s1"].is_not_null().sum())
emp = emp.filter(pl.col("s1").is_not_null()).join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), pl.col("addr").alias("s1_addr")), on="s1")
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
s1 = s1.with_columns(norm("name").alias("nn"))
emp = emp.with_columns(norm("name").alias("cn"), norm("s1_name").alias("s1n"), norm("s1_addr").alias("s1a"))
# namesake count of the true S1 by normalized name within the country
cnt = s1.group_by(["nn", "country"]).len().rename({"len": "n_same"})
emp = emp.join(cnt, left_on=["s1n", "country"], right_on=["nn", "country"], how="left")
print(" namesake group size of true S1 (normalized name):", emp.group_by("n_same").len().sort("n_same").head(12).to_dicts())
print(" copy name == S1 name (normalized):", (emp["cn"] == emp["s1n"]).mean())
# (a) extra tokens in copy name
rows = emp.select("cn", "s1n", "s1a", "n_same").to_dicts()
extra_in_addr = 0; extra_any = 0; tokc = collections.Counter(); tok_in_addr = collections.Counter()
for r in rows:
    ct = set(r["cn"].split()); st = set(r["s1n"].split()); at = set(r["s1a"].split())
    ex = ct - st
    if ex:
        extra_any += 1
        if ex & at:
            extra_in_addr += 1
            for w in ex & at:
                tok_in_addr[w] += 1
        for w in ex:
            tokc[w] += 1
print(f" copies with extra tokens: {extra_any}/{len(rows)}; extra token found in S1 address: {extra_in_addr}")
print(" top extra tokens:", tokc.most_common(40))
print(" top extra tokens that are in S1 addr:", tok_in_addr.most_common(30))
# (b) namesake S1s: raw name identical? show example groups
ex = emp.filter(pl.col("n_same") >= 3).head(8)
for r in ex.to_dicts():
    grp = s1.filter((pl.col("nn") == r["s1n"]) & (pl.col("country") == r["country"]))
    print("\n COPY", r["entity_id"], "|", r["name"], "| true", r["s1"])
    print(grp.select("entity_id", "name", "addr"))
# raw-name variety within namesake groups
g = s1.group_by(["nn", "country"]).agg(pl.len().alias("n"), pl.col("name").n_unique().alias("raw_variants")).filter(pl.col("n") > 1)
print("\n namesake groups:", g.height, "with >1 raw variant:", (g["raw_variants"] > 1).sum())
