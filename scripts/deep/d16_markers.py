"""Deep dive 16: copy markers in train. Rates of name/address perturbation markers among copies vs non-copies (noise
records with no S1 in the full truth). A marker that never appears on noise flags 'copy of some entity'.

uv run --no-project --with polars==1.44.2 python scripts/deep/d16_markers.py
"""
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
tp = pl.read_parquet("local_data/train_pairs.parquet")
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (2, 3)]).with_columns(pl.col("entity_id").is_in(tp["m"].implode()).alias("copy"))
n, a = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
M = {
    "bracket": n.str.contains(r"[\[\(\{]"), "dspace": n.str.contains("  "), "dba": n.str.contains(r"(?i)\bd\.?b\.?a\b|trading as|formerly|aka\b|t/a\b"),
    "handle": n.str.contains(r"(?i)\.com\b|^#|^@|www\."), "prefix_punct": n.str.contains(r"^\W"), "upper": n.str.contains(r"^[^a-z]*[A-Z][^a-z]*$"),
    "lower": n.str.contains(r"^[^A-Z]*[a-z][^A-Z]*$"), "accent": n.str.contains(r"[À-ÿ]"), "digit_in_name": n.str.contains(r"\d"),
    "addr_empty": a.str.strip_chars() == "", "addr_upper": a.str.contains(r"^[^a-z]*[A-Z][^a-z]*$"), "addr_hash": a.str.contains(r"#|Nº|N°|No\.? ?\d"),
    "addr_zero_pad": a.str.contains(r"(^|\s)0\d"), "one_token": ~n.str.strip_chars().str.contains(" "),
}
x = rec.with_columns(**M)
rows = []
for k in M:
    t = x.group_by("country", "copy").agg(pl.col(k).mean().alias("r"))
    for c, cp, r in t.iter_rows():
        rows.append((k, c, cp, r))
d = pl.DataFrame(rows, schema=["marker", "country", "copy", "rate"], orient="row").pivot(on="copy", index=["marker", "country"], values="rate")
d = d.rename({"true": "copy", "false": "noise"}).with_columns((pl.col("copy") / (pl.col("noise") + 1e-6)).round(1).alias("lift"), pl.col("copy").round(4), pl.col("noise").round(4))
print(d.sort("marker", "country"))
print("noise share:", x.group_by("country").agg((~pl.col("copy")).mean()).rows())
print(x.filter(~pl.col("copy")).sample(25, seed=2).select("entity_id", "business_name", "business_address", "country"))
