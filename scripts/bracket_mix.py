"""Decoy share of French same-house type-word swaps from formatting markers. Copy noise brackets the new word ("[Club]",
"(Club)") in ~4.5% of French copies vs ~1.2% of decoys; lowercase names and legal-form changes are other markers.
References: generic-word swaps at the same house (copies), type-word swaps at a decoy offset (sibling decoys).
Mixture: d = (m_copy - m_obs) / (m_copy - m_decoy) per marker. Classes split by fr12 / fr7 acceptance.

python scripts/bracket_mix.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie", "service"}
DW = {"holding", "participations", "distribution", "international", "groupe", "developpement", "snc"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def abbrev(x, y):
    if not x or not y or x[0] != y[0] or len(x) >= len(y):
        return False
    it = iter(y)
    return all(ch in it for ch in x)


te = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name", "country", "src", "addr_n", "house"]) \
       .with_row_index("rid").filter(pl.col("country") == "France")
rid = te["rid"].to_list(); src = te["src"].to_list(); nm = te["business_name"].to_list(); hs = te["house"].to_list()
raw = [set(re.findall(r"[a-z0-9]+", norm(n))) for n in nm]
nt = [t - LEGF - STOP for t in raw]
lf = [" ".join(sorted(t & LEGF)) for t in raw]
at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in te["addr_n"].to_list()]
street = [" ".join(sorted(t - {h})) if h else " ".join(sorted(t)) for t, h in zip(at, hs)]
hn = [int(h) if h and str(h).isdigit() else None for h in hs]
lower = [n == n.lower() for n in nm]
by = {}
for i in range(len(rid)):
    if src[i] == 1 and street[i]:
        by.setdefault(street[i], []).append(i)
rows = []
for i in range(len(rid)):
    if src[i] == 1 or not street[i] or raw[i] & DW - GEN:
        continue
    A = nt[i]
    for j in by.get(street[i], []):
        B = nt[j]
        if not (len(A - B) == 1 and len(B - A) == 1 and A & B):
            continue
        wi, wo = next(iter(A - B)), next(iter(B - A))
        if hn[i] is None or hn[j] is None:
            continue
        dh = hn[i] - hn[j]
        hrel = "same" if dh == 0 else ("offset" if dh in OFFS else None)
        if hrel is None:
            continue
        near = fuzz.ratio(wi, wo) >= 70 or abbrev(wi, wo) or abbrev(wo, wi) or wi.startswith(wo[:4]) or wo.startswith(wi[:4])
        kind = "generic" if wi in GEN else ("near" if near else "type")
        br = bool(re.search(r"[\[\(]\s*" + re.escape(wi), norm(nm[i])))
        rows.append((rid[i], rid[j], kind, hrel, br, lower[i], lf[i] == lf[j], src[i]))
x = pl.DataFrame(rows, schema=["q", "s1", "kind", "hrel", "bracket", "lower", "legal_same", "src"], orient="row") \
      .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
eid = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
            .filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


for tag in ("fr7", "fr12"):
    x = x.join(accepted(f"/work/out_{tag}").with_columns(pl.lit(True).alias(f"acc_{tag}")), on=["q", "s1"], how="left") \
         .with_columns(pl.col(f"acc_{tag}").fill_null(False))
x = x.with_columns(pl.when(pl.col("acc_fr12")).then(pl.lit("kept")).when(pl.col("acc_fr7")).then(pl.lit("removed"))
                   .otherwise(pl.lit("rejected")).alias("status"))
t = x.group_by("kind", "hrel", "status").agg(pl.len().alias("n"), pl.col("bracket").mean().round(4).alias("brk"),
                                              pl.col("lower").mean().round(4).alias("low"), pl.col("legal_same").mean().round(4).alias("leg_same")) \
     .filter(pl.col("n") >= 30).sort("kind", "hrel", "status")
print(t)
t2 = x.group_by("kind", "hrel").agg(pl.len().alias("n"), pl.col("bracket").mean().round(4).alias("brk"), pl.col("lower").mean().round(4).alias("low"),
                                     pl.col("legal_same").mean().round(4).alias("leg_same")).sort("kind", "hrel")
print(t2)
ref = {r[0] + "|" + r[1]: r for r in t2.rows()}
mc, md = ref["generic|same"], ref["type|offset"]
print(f"copy ref generic|same brk {mc[3]} low {mc[4]} leg {mc[5]}; decoy ref type|offset brk {md[3]} low {md[4]} leg {md[5]}")
for r in t.filter((pl.col("kind") == "type") & (pl.col("hrel") == "same")).rows():
    est = [(mc[k] - r[k + 1]) / (mc[k] - md[k]) if mc[k] != md[k] else None for k in (3, 4, 5)]
    print(f"type|same {r[2]:9} n {r[3]:6}  d(bracket) {est[0]:.2f}  d(lower) {est[1]:.2f}  d(legal) {est[2]:.2f}")
x.write_parquet("/work/lead/fr_bracket_mix.parquet")
