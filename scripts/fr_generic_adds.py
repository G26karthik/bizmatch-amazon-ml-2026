"""Unassigned France records whose best S1 (model) differs only by a generic copy word (swap to / add of fils, cie,
associes, services, developpement, groupe, france, frs, compagnie, etablissements, societe...), legal form kept, same house
number, street similar (typos, abbreviations). Reports counts by probability band and address closeness, with samples.
Writes the candidate additions for the fr8 chain.

python scripts/fr_generic_adds.py /work/out_fr8
"""
import re
import sys
import unicodedata

import polars as pl
from rapidfuzz import fuzz

out = sys.argv[1]
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"}
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "e", "r"}
ABBR = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "pl": "place", "ch": "chemin", "rte": "route",
        "imp": "impasse", "all": "allee", "q": "quai", "crs": "cours", "sq": "square", "pass": "passage", "psg": "passage", "st": "saint", "ste": "sainte"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def ntok(s):
    t = re.findall(r"[a-z0-9]+", norm(s).replace(".", ""))
    return set(t) - STOP


def atok(s):
    return {ABBR.get(t, t) for t in re.findall(r"[a-z0-9]+", norm(s).replace(".", " "))} - STOP


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
fr = rec.filter(pl.col("country") == "France")
d = tb.join(fr.filter(pl.col("src") != 1).select(pl.col("rid").alias("q")), on="q").join(acc, on="q", how="anti")
nm = rec["business_name"].to_list(); ra = rec["business_address"].to_list(); hs = rec["house"].to_list()
rows = []
for q, s, p in d.iter_rows():
    A, B = ntok(nm[q]), ntok(nm[s])
    la, lb = A & LEG, B & LEG
    A2, B2 = A - LEG, B - LEG
    extra, miss = A2 - B2, B2 - A2
    if not extra or not extra <= GEN or len(extra) > 1 or len(miss) > 1 or not (A2 & B2):
        continue
    if la != lb:
        continue
    if not ra[q] or not hs[q] or hs[q] != hs[s]:
        continue
    ta, tb_ = atok(ra[q]), atok(ra[s])
    j = len(ta & tb_) / max(len(ta | tb_), 1)
    sim = fuzz.token_set_ratio(norm(ra[q]), norm(ra[s]))
    rows.append((q, s, p, "swap" if miss else "add", round(j, 2), sim))
x = pl.DataFrame(rows, schema=["q", "s1", "prob", "op", "jacc", "sim"], orient="row")
x = x.with_columns(pl.col("prob").cut([0.05, 0.2, 0.5, 0.8]).alias("pb"), (pl.col("sim") >= 85).alias("addr_close"))
print("unassigned generic-edit records with same house:", x.height)
print(x.group_by("pb", "addr_close").len().sort("pb", "addr_close"))
for row in x.filter(pl.col("addr_close")).sample(n=min(16, x.height), seed=4).iter_rows(named=True):
    print(f"  p={row['prob']:.2f} {row['op']} sim={row['sim']}\n     REC {nm[row['q']]} | {ra[row['q']]}\n     S1  {nm[row['s1']]} | {ra[row['s1']]}")
x.write_parquet("/work/dive/fr_generic_adds.parquet")
