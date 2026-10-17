"""Deep dive 21: unassigned test records carrying copy-only markers (d/b/a-type alias, web handle, one-token name,
empty address) per 1k S1s by country, fr19. Train: these markers are 8-180x more frequent on copies than on noise,
and test has no hidden-S1 copies -> an unassigned marked record is (mostly) a missed copy of a visible S1.

uv run --no-project --with polars==1.44.2 python scripts/deep/d21_marked_unassigned.py [base]
"""
import sys

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(60)
base = sys.argv[1] if len(sys.argv) > 1 else "output/v28q_fr19"
rec = pl.concat([pl.read_parquet(f"local_data/{sp}_source{s}.parquet").with_columns(pl.lit(sp).alias("split")) for sp in ("test",) for s in (1, 2, 3)])
g = pl.read_csv(f"{base}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
n, a = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
r = rec.filter(~pl.col("entity_id").str.starts_with("S1")).with_columns(
    pl.col("entity_id").is_in(assigned.implode()).alias("asg"),
    n.str.contains(r"(?i)\bd\.?b\.?a\b|trading as|formerly|aka\b|t/a\b").alias("dba"),
    n.str.contains(r"(?i)\.com\b|^#|^@|www\.").alias("handle"),
    (~n.str.strip_chars().str.contains(" ")).alias("one_token"),
    (a.str.strip_chars() == "").alias("empty"))
ns1 = rec.filter(pl.col("entity_id").str.starts_with("S1")).group_by("country").len().rename({"len": "S1s"})
rows = []
for mk in ("dba", "handle", "one_token", "empty"):
    t = r.filter(pl.col(mk)).group_by("country").agg(pl.len().alias("n"), (~pl.col("asg")).sum().alias("unassigned")).join(ns1, on="country")
    rows.append(t.with_columns(pl.lit(mk).alias("marker"), (pl.col("unassigned") / pl.col("S1s") * 1000).round(2).alias("unasg_per1k"),
                               (pl.col("unassigned") / pl.col("n")).round(4).alias("unasg_share")))
print(pl.concat(rows).select("marker", "country", "n", "unassigned", "unasg_share", "unasg_per1k").sort("marker", "country"))
fr = r.filter((pl.col("country") == "France") & ~pl.col("asg") & (pl.col("dba") | pl.col("handle")))
print("France unassigned d/b/a or handle examples:")
print(fr.sample(min(25, fr.height), seed=3).select("entity_id", "business_name", "business_address"))
