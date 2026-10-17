"""Generator dive 6 (worker): record-level STYLE flags (formatting only, no pairing) of true copies vs decoys, per country x source.
If the decoy generator formats records differently from the copy generator, style alone separates them.
needs decoys.parquet (g0_samples.py)
"""
import polars as pl

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230)
R = "/work/run4/train/records.parquet"
dec = pl.read_parquet("/work/dive/decoys.parquet").select(pl.col("rid").cast(pl.UInt32), pl.lit(True).alias("decoy"))
r = pl.scan_parquet(R).select("business_name", "business_address", "src", "country", "addr_empty").with_row_index("rid") \
      .filter(pl.col("src") > 1).collect()
r = r.join(dec, on="rid", how="left").with_columns(pl.col("decoy").fill_null(False))
n, a = pl.col("business_name"), pl.col("business_address")
has_alpha = lambda e: e.str.contains(r"[A-Za-z]")  # noqa: E731
f = r.select(
    "country", "src", "decoy", "addr_empty",
    ((n == n.str.to_uppercase()) & has_alpha(n)).alias("name_UPPER"),
    ((n == n.str.to_lowercase()) & has_alpha(n)).alias("name_lower"),
    (n == n.str.to_titlecase()).alias("name_Title"),
    n.str.contains("  ").alias("name_dblspace"),
    (n != n.str.strip_chars()).alias("name_padspace"),
    n.str.contains(r"[\(\[\{]").alias("name_bracket"),
    n.str.contains(r"(?i)\.com|www\.|@|\|").alias("name_web"),
    n.str.contains(r"(?i)\b(formerly|fka|aka|dba|f/k/a|a/k/a|d/b/a)\b").alias("name_aka"),
    n.str.contains(r"[^\x00-\x7f]").alias("name_nonascii"),
    n.str.contains(r"[àáâäèéêëìíîïòóôöùúûüçñÀ-ÿ]").alias("name_accent"),
    n.str.contains(r"#|>>|\*|~").alias("name_symbol"),
    n.str.contains(r"\d").alias("name_digit"),
    n.str.contains(r"(?i)^(mr|mrs|ms|smt|shri|sri|dr|m/s)\.?\s").alias("name_honorific"),
    ((a == a.str.to_uppercase()) & has_alpha(a)).alias("addr_UPPER"),
    (a == a.str.to_titlecase()).alias("addr_Title"),
    a.str.contains("  ").alias("addr_dblspace"),
    a.str.contains("#").alias("addr_hash"),
    a.str.contains(r"(?i)\b(po box|pmb)\b").alias("addr_pobox"),
    a.str.contains(r"(?i)\bn/a\b").alias("addr_na"),
    a.str.contains(r"[^\x00-\x7f]").alias("addr_nonascii"),
    a.str.contains(r"^[A-Za-z ]+,").alias("addr_starts_word"),
    a.str.contains(r"^\s*\d").alias("addr_starts_digit"),
    a.str.contains(r",\s*[A-Z]{2}\s*$").alias("addr_ends_2letter"),
    a.str.contains(r"\b0\d+").alias("addr_leading_zero"),
    a.str.contains(r"\d+[A-Za-z]\b").alias("addr_num_letter"),
)
flags = [c for c in f.columns if c not in ("country", "src", "decoy", "addr_empty")]
g = f.group_by("country", "src", "decoy").agg(pl.len(), *[pl.col(c).mean().round(4) for c in flags]).sort("country", "src", "decoy")
long = g.unpivot(index=["country", "src", "decoy"], on=flags).with_columns(pl.format("{}-S{}-{}", "country", "src", pl.when(pl.col("decoy")).then(pl.lit("dec")).otherwise(pl.lit("cop"))).alias("k"))
w = long.pivot(on="k", index="variable", values="value")
print(g.select("country", "src", "decoy", "len"))
print(w)
# match rate of records having each flag (per country x src)
print("\nmatch rate (share true copy) of records with flag=1 vs flag=0:")
rows = []
for c in flags:
    t = f.group_by("country", "src", c).agg(pl.len(), (~pl.col("decoy")).mean().round(4).alias("match_rate"))
    for d in t.iter_rows(named=True):
        rows.append(dict(flag=c, country=d["country"], src=d["src"], val=d[c], n=d["len"], match_rate=d["match_rate"]))
m = pl.DataFrame(rows).filter(pl.col("val")).with_columns(pl.format("{}-S{}", "country", "src").alias("k"))
print(m.pivot(on="k", index="flag", values="match_rate"))
print(m.pivot(on="k", index="flag", values="n"))
