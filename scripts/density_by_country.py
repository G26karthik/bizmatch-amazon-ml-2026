"""Records per S1 by country in test: all S2/S3 records, accepted (fr7 / fr12), unassigned, decoy-word records.
If the non-copy density were equal across countries, France copies per S1 = records - non-copies.

python scripts/density_by_country.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
DW = {"holding", "participations", "distribution", "international", "groupe", "developpement", "snc", "southside", "eastgate",
      "northside", "midtown", "greater"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name"])
rec = rec.with_columns(pl.Series("dw", [bool(set(re.findall(r"[a-z0-9]+", norm(n))) & DW) for n in rec["business_name"].to_list()]))
ns1 = dict(rec.filter(pl.col("src") == 1).group_by("country").len().iter_rows())
eid_c = rec.select("entity_id", "country", "src", "dw")
rows = []
for tag in ("fr7", "fr12"):
    g = pl.read_csv(f"/work/out_{tag}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls() \
           .filter(pl.col("matched_entity_ids") != "").select(pl.col("matched_entity_ids").alias("entity_id")).unique()
    a = acc.join(eid_c, on="entity_id")
    for c, n in a.group_by("country").len().iter_rows():
        rows.append((tag, c, "accepted", n / ns1[c]))
    for c, n in a.filter(pl.col("dw")).group_by("country").len().iter_rows():
        rows.append((tag, c, "accepted_dw", n / ns1[c]))
    e = g.with_columns(pl.col("matched_entity_ids").fill_null("")).join(eid_c.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
    for c, n in e.filter(pl.col("matched_entity_ids") == "").group_by("country").len().iter_rows():
        rows.append((tag, c, "empty_S1_rate", n / ns1[c]))
    un = eid_c.filter(pl.col("src") != 1).join(acc, on="entity_id", how="anti")
    for c, n in un.group_by("country").len().iter_rows():
        rows.append((tag, c, "unassigned", n / ns1[c]))
    for c, n in un.filter(~pl.col("dw")).group_by("country").len().iter_rows():
        rows.append((tag, c, "unassigned_non_dw", n / ns1[c]))
for c, n in rec.filter(pl.col("src") != 1).group_by("country").len().iter_rows():
    rows.append(("all", c, "records", n / ns1[c]))
for c, n in rec.filter((pl.col("src") != 1) & pl.col("dw")).group_by("country").len().iter_rows():
    rows.append(("all", c, "decoy_word_records", n / ns1[c]))
x = pl.DataFrame(rows, schema=["tag", "country", "what", "per_S1"], orient="row").with_columns(pl.col("per_S1").round(4))
print(x.pivot(on="country", index=["tag", "what"], values="per_S1"))
print("S1 counts", ns1)
