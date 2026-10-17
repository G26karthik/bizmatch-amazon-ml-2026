"""Vocabulary check per country: tokens frequent in S2/S3 records (names, addresses) but rare in S1 records reveal generator
noise operations (abbreviations, translations, decoy words). Also splits by our assignment (accepted vs unassigned).

python scripts/fr_vocab.py /work/out_fr7
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(eid.rename({"entity_id": "matched_entity_ids"}), on="matched_entity_ids").select("rid").unique()
rec = rec.with_columns(pl.col("rid").is_in(acc["rid"].implode()).alias("acc"))
for country in ("France", "US"):
    r = rec.filter(pl.col("country") == country)
    for field in ("business_name", "business_address"):
        cnt = {}
        for key, sub in (("s1", r.filter(pl.col("src") == 1)), ("acc", r.filter((pl.col("src") != 1) & pl.col("acc"))),
                         ("un", r.filter((pl.col("src") != 1) & ~pl.col("acc")))):
            c = Counter()
            for s in sub[field].to_list():
                c.update(set(re.findall(r"[a-z]+", norm(s))))
            cnt[key] = (c, sub.height)
        toks = set(cnt["acc"][0]) | set(cnt["un"][0])
        rows = []
        for t in toks:
            a, u, s = cnt["acc"][0][t], cnt["un"][0][t], cnt["s1"][0][t]
            if a + u < 300:
                continue
            fs = s / cnt["s1"][1]
            fo = (a + u) / (cnt["acc"][1] + cnt["un"][1])
            rows.append((t, s, a, u, round(fo / max(fs, 1e-7), 2), round(u / max(a + u, 1), 3)))
        df = pl.DataFrame(rows, schema=["tok", "n_s1", "n_acc", "n_unassigned", "ratio_other_vs_s1", "unassigned_share"], orient="row")
        print(f"\n=== {country} {field}: S1 {cnt['s1'][1]}, accepted {cnt['acc'][1]}, unassigned {cnt['un'][1]}; tokens over-represented outside S1")
        print(df.filter(pl.col("ratio_other_vs_s1") >= 3).sort("ratio_other_vs_s1", descending=True).head(45))
