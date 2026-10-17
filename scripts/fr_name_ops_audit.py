"""Audit accepted France pairs of a submission by name-difference pattern (name key = tokens minus legal/stop words):
ADD one word (generic / vocab / non-word), DROP one word, SWAP (generic / vocab / non-word, near-spelling), 2+ differences,
identical; with address relation, bracket share of the new word and decoy estimate d = (0.0418 - brk)/(0.0418 - 0.0090).

python scripts/fr_name_ops_audit.py /work/out_fr13 [--unaccepted]   (--unaccepted: best-S1 pairs with prob > 0.1 NOT accepted)
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
out = sys.argv[1]
BC, BD = 0.0418, 0.0090
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
LEG = LEGF | {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie", "service"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); hs = rec["house"].to_list(); ad = rec["addr_n"].to_list()
fr = rec.filter(pl.col("country") == "France")
voc1 = Counter(w for n in fr.filter(pl.col("src") == 1)["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", norm(n))))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1") \
       .join(fr.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1")), on="s1", how="semi")
n_acc = dict(acc.group_by("s1").len().iter_rows())
UNACC = "--unaccepted" in sys.argv
if UNACC:
    tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")            .filter(pl.col("prob") > 0.1).join(fr.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1")), on="s1", how="semi")
    acc = tb.join(acc, on=["q", "s1"], how="anti").join(acc.select("q"), on="q", how="anti").select("q", "s1")


def wkind(w, other=None):
    if w in GEN:
        return "generic"
    if other and (fuzz.ratio(w, other) >= 70 or w.startswith(other[:4]) or other.startswith(w[:4])):
        return "near"
    return "vocab" if voc1.get(w, 0) > 30 else "nonword"


rows = []
for q, s in zip(acc["q"].to_list(), acc["s1"].to_list()):
    rq, rs = set(re.findall(r"[a-z0-9]+", norm(nm[q]))), set(re.findall(r"[a-z0-9]+", norm(nm[s])))
    A, B = rq - LEG, rs - LEG
    new = None
    if A == B:
        pat = "identical_key"
    elif B < A and len(A - B) == 1:
        new = next(iter(A - B)); pat = "ADD_" + wkind(new)
    elif A < B and len(B - A) == 1:
        pat = "DROP"
    elif len(A - B) == 1 and len(B - A) == 1:
        new = next(iter(A - B)); pat = "SWAP_" + wkind(new, next(iter(B - A)))
    else:
        pat = f"diff{min(len(A ^ B), 6)}"
    arel = "no_addr" if not ad[q] or not ad[s] else ("same_house" if hs[q] and hs[q] == hs[s] else ("no_house" if not hs[q] or not hs[s] else "diff_house"))
    br = bool(new) and bool(re.search(r"[\[\(]\s*" + re.escape(new), norm(nm[q])))
    rows.append((q, s, pat, arel, br, (rq & LEGF) == (rs & LEGF), n_acc.get(s, 0) == (0 if UNACC else 1), new))
x = pl.DataFrame(rows, schema=["q", "s1", "pat", "arel", "bracket", "legal_same", "single", "new"], orient="row")
t = x.group_by("pat").agg(pl.len().alias("n"), pl.col("bracket").mean().round(4).alias("brk"), pl.col("legal_same").mean().round(3).alias("leg"),
                          pl.col("single").mean().round(3).alias("single"), (pl.col("arel") == "same_house").mean().round(3).alias("same_h")) \
      .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).sort("n", descending=True)
print(f"accepted France pairs {x.height}"); print(t)
print("\nADD_vocab / SWAP_vocab by address:")
print(x.filter(pl.col("pat").str.starts_with("SWAP") | pl.col("pat").str.starts_with("ADD")).group_by("pat", "arel").agg(pl.len(), pl.col("bracket").mean().round(4)).sort("pat", "arel"))
for p in (("SWAP_nonword", "SWAP_generic", "identical_key") if UNACC else ("ADD_vocab", "diff2")):
    print(f"\nexamples {p}")
    for q, s in x.filter(pl.col("pat") == p).sample(n=min(12, x.filter(pl.col("pat") == p).height), seed=3).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:44} | {nm[q]!r:44} | {ad[s]} | {ad[q]}")
x.write_parquet("/work/lead/fr_name_ops_audit" + ("_unacc" if UNACC else "") + ".parquet")
