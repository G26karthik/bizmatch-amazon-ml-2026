"""Refine and cap-test the France initials / pseudo-name adds (init_pseudo_rule.py -> fr_init_pseudo_adds.parquet).
PSEUDO_BRAND: every name-key token of the record is absent from French S1 names (a generated pseudo brand, e.g. Zetacalo);
PSEUDO_VOCAB: the rest (real French words -> another business at the same address). Cap test on unassigned vs accepted
single-S1-address records of each class (copies vanish at the S1's source cap; the model's cap awareness does not apply to
unassigned records).

python scripts/init_pseudo_cap.py /work/out_fr13
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
out = sys.argv[1]
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


te = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name", "src", "country"]).with_row_index("rid") \
       .with_columns(pl.col("rid").cast(pl.UInt32))
nm = te["business_name"].to_list(); srcd = dict(zip(te["rid"].to_list(), te["src"].to_list()))
fr1 = te.filter((pl.col("country") == "France") & (pl.col("src") == 1))
voc1 = Counter(w for n in fr1["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", norm(n))))
adds = pl.read_parquet("/work/lead/fr_init_pseudo_adds.parquet")
cls2 = []
for q, c in zip(adds["q"].to_list(), adds["cls"].to_list()):
    if c == "INIT":
        cls2.append("INIT")
        continue
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
    cls2.append("PSEUDO_BRAND" if A and all(voc1.get(w, 0) == 0 and len(w) >= 4 for w in A) else "PSEUDO_VOCAB")
adds = adds.with_columns(pl.Series("cls2", cls2))
print("unassigned adds by refined class:", adds.group_by("cls2").len().rows())
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
acc = acc.with_columns(pl.Series("src", [srcd[q] for q in acc["q"].to_list()], dtype=pl.Int8))
oth = acc.group_by("s1", "src").agg(pl.len().alias("n_other"))
s1f = fr1.select(pl.col("rid").alias("s1"))
den = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
d23 = den.filter(pl.col("n_other").is_between(2, 3)).height
d4 = den.filter(((pl.col("src") == 2) & (pl.col("n_other") == 4)) | ((pl.col("src") == 3) & (pl.col("n_other").is_between(4, 5)))).height
z = adds.with_columns(pl.Series("src", [srcd[q] for q in adds["q"].to_list()], dtype=pl.Int8)).join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
# adds were filtered to S1s under the cap, so compare 2-3 others with the last slot below the cap (4 for S2, 4-5 for S3)
t = z.group_by("cls2").agg(pl.col("n_other").is_between(2, 3).sum().alias("n23"),
                           (((pl.col("src") == 2) & (pl.col("n_other") == 4)) | ((pl.col("src") == 3) & pl.col("n_other").is_between(4, 5))).sum().alias("n_near_cap")) \
      .with_columns((pl.col("n23") / d23 * 1000).round(3).alias("r23"), (pl.col("n_near_cap") / d4 * 1000).round(3).alias("r_near_cap")) \
      .with_columns((pl.col("r_near_cap") / pl.col("r23")).round(3).alias("ratio"))
print(f"slots: n23 {d23}, near-cap {d4}"); print(t)
for c in ("PSEUDO_BRAND", "PSEUDO_VOCAB"):
    print(f"examples {c}")
    for q, s in adds.filter(pl.col("cls2") == c).sample(n=10, seed=8).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:44} | {nm[q]!r}")
adds.write_parquet("/work/lead/fr_init_pseudo_adds2.parquet")
