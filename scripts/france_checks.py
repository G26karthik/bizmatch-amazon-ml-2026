"""Label-free France checks for the France additions.

1. Gate v2 false-fire rate: France S1 businesses are unique, so two S1s at the same exact address are different
   businesses. How often would the v2 rule (shared distinctive word, no decoy word) fire on such S1-S1 pairs?
2. Empty-S1 rate per country for candidate outputs: the generator gives 5.58% of S1s no copy in every country; US/India
   predictions sit at ~5.75% (truth + misses). A lower France rate means matches given to true singletons.

python scripts/france_checks.py /work/out_lite7 /work/out_fr4 ...
"""
import re
import sys
import unicodedata
from collections import Counter
from itertools import combinations

import polars as pl

STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e"}
DECOY = {"holding", "participations", "international", "distribution", "groupe", "developpement", "france"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", s)) - STOP


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
fr1 = rec.filter((pl.col("src") == 1) & (pl.col("country") == "France"))
df = Counter(t for n in fr1["business_name"].to_list() for t in toks(n))
cut = 0.002 * fr1.height
g = fr1.filter(pl.col("house").is_not_null() & (pl.col("house") != "") & (pl.col("addr_n") != "")).group_by("addr_n").agg(pl.col("business_name")) \
       .filter(pl.col("business_name").list.len() > 1)
n = fire = anyw = 0
for names in g["business_name"].to_list():
    for x, y in combinations(names[:6], 2):
        tx, ty = toks(x), toks(y)
        n += 1
        anyw += bool(tx & ty)
        dec = bool((tx - ty) & DECOY) or bool((ty - tx) & DECOY)
        fire += (not dec) and any(df.get(t, 0) <= cut for t in tx & ty)
print(f"1. France S1-S1 pairs at the same exact address: {n}; share any word {anyw / max(n, 1):.4f}; "
      f"gate-v2 rule fires {fire} ({fire / max(n, 1):.4f})")

eid = rec.select("rid", "entity_id")
s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country")
for path in sys.argv[1:]:
    gdf = pl.read_csv(f"{path}/matching_results.tsv", separator="\t")
    p = gdf.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
           .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id").select("s1").unique()
    nonempty = s1.join(p, on="s1", how="semi").group_by("country").len().rename({"len": "nonempty"})
    tot = s1.group_by("country").len().rename({"len": "total"})
    r = tot.join(nonempty, on="country").with_columns((1 - pl.col("nonempty") / pl.col("total")).round(5).alias("empty_rate")).sort("country")
    print(f"2. {path}: empty S1 rate {dict(zip(r['country'].to_list(), r['empty_rate'].to_list()))}")
