"""Deep dive 8: unassigned empty-address test records whose name key matches no S1 key exactly (n_ns = 0).
Are they in the uncertain zone (unc_test, 0.02<p<0.98)? Top candidate prob, number of candidates, examples.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d08_fuzzy_empty.py [country] [n_examples]
"""
import re
import sys
import unicodedata

import polars as pl

ctry = sys.argv[1] if len(sys.argv) > 1 else "US"
nex = int(sys.argv[2]) if len(sys.argv) > 2 else 20

def key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return " ".join(re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch))))

rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"), pl.col("entity_id").str.slice(0, 2).alias("src"))
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list()
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(e for m in g["matched_entity_ids"].fill_null("").to_list() for e in m.split(",") if e)
s1k = set(zip(rec.filter(pl.col("src") == "S1")["country"].to_list(), rec.filter(pl.col("src") == "S1")["k"].to_list()))
r = rec.filter((pl.col("src") != "S1") & (pl.col("country") == ctry) & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
r = r.filter(~pl.col("entity_id").is_in(list(assigned)))
r = r.filter(pl.Series([(ctry, k) not in s1k for k in r["k"].to_list()]))
print(ctry, "unassigned empty-address records with no exact-key S1:", r.height)
unc = pl.read_parquet("local_data/llm/unc_test.parquet")
u = unc.filter(pl.col("q").is_in(r["rid"].implode()))
top = u.sort("prob", descending=True).group_by("q").agg(pl.col("prob").first().alias("p1"), pl.col("prob").slice(1, 1).first().alias("p2"), pl.len().alias("ncand"), pl.col("s1").first().alias("s1"))
print("in uncertain zone:", top.height, "of", r.height)
print("p1 quantiles:", [round(top["p1"].quantile(q), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
print("p2 quantiles:", [round(top["p2"].fill_null(0).quantile(q), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
print("ncand dist:", top.group_by(pl.col("ncand").clip(1, 6)).len().sort("ncand").rows())
for q, p1, p2, nc in top.sample(min(nex, top.height), seed=2).select("q", "p1", "p2", "ncand").iter_rows():
    print(f"\nREC {eid[q]} {nm[q]!r}  p1={p1:.3f} p2={p2 if p2 is None else round(p2, 3)} ncand={nc}")
    for s, p in u.filter(pl.col("q") == q).sort("prob", descending=True).head(5).select("s1", "prob").iter_rows():
        print(f"     {p:.3f} {eid[s]} {nm[s]!r} | {ad[s]}")
notin = r.filter(~pl.col("rid").is_in(top["q"].implode()))
print("\nNOT in uncertain zone (examples):")
for q in notin.sample(min(10, notin.height), seed=1)["rid"].to_list():
    print(f"   {eid[q]} {nm[q]!r}")
