"""Cap test on France swap subsets (records on the S1 street, same house number, one-word name swap):
  E  exact address, generic word            (copy reference)
  A  exact address, type word -> other type word (fr10 set)
  B  exact address, near-spelling swap between two S1-vocabulary words (sport -> sportive)
  T  exact address, near-spelling to a non-vocabulary word (typo copy reference)
  C  same house, street edited (typo), type -> type
  D  same house, street edited, generic word
Rate per 1k S1 by other accepted records in the same source (cap S2=5, S3=6); copies vanish at the cap.

python scripts/cap_test2.py /work/out_fr10
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

out = sys.argv[1]
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def abbrev(x, y):
    if not x or not y or x[0] != y[0] or len(x) >= len(y):
        return False
    it = iter(y)
    return all(ch in it for ch in x)


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).filter(pl.col("country") == "France")
nt = {r: set(re.findall(r"[a-z0-9]+", norm(n))) - LEG for r, n in zip(rec["rid"].to_list(), rec["business_name"].to_list())}
at = {r: set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for r, a in zip(rec["rid"].to_list(), rec["addr_n"].to_list())}
hs = dict(zip(rec["rid"].to_list(), rec["house"].to_list()))
src = dict(zip(rec["rid"].to_list(), rec["src"].to_list()))
s1ids = rec.filter(pl.col("src") == 1)["rid"].to_list()
voc = Counter(w for s in s1ids for w in nt[s])
# candidate pairs: best-S1 pairs + accepted pairs
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1")
pool = pl.concat([tb, acc]).unique().join(rec.select(pl.col("rid").alias("s1")).filter(pl.col("s1").is_in(s1ids)), on="s1")
pool = pool.filter(pl.col("q").is_in(list(nt.keys())))
cats = []
for q, s in zip(pool["q"].to_list(), pool["s1"].to_list()):
    A, B = nt[q], nt[s]
    c = None
    if len(A - B) == 1 and len(B - A) == 1 and A & B and hs.get(q) and hs.get(q) == hs.get(s):
        wi, wo = next(iter(A - B)), next(iter(B - A))
        exact = at[q] == at[s]
        streetq, streets = at[q] - {hs[q]}, at[s] - {hs[s]}
        close = not exact and len(streetq ^ streets) <= 2 and fuzz.token_set_ratio(" ".join(sorted(streetq)), " ".join(sorted(streets))) >= 80
        if exact or close:
            near = fuzz.ratio(wi, wo) >= 70 or abbrev(wi, wo) or abbrev(wo, wi) or wi.startswith(wo[:4]) or wo.startswith(wi[:4])
            if wi in GEN:
                c = "E_exact_generic" if exact else "D_edit_generic"
            elif near:
                c = ("B_exact_near_vocab" if voc[wi] >= 50 and not abbrev(wi, wo) else "T_exact_near_typo") if exact else None
            else:
                c = "A_exact_type" if exact else "C_edit_type"
    cats.append(c)
pool = pool.with_columns(pl.Series("cat", cats)).filter(pl.col("cat").is_not_null())
pool = pool.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
pool = pool.with_columns(pl.Series("src", [src[q] for q in pool["q"].to_list()]))
catq = pool.select("q").unique()
oc = acc.join(catq, on="q", how="anti").with_columns(pl.Series("src", [src.get(q, 0) for q in acc.join(catq, on="q", how="anti")["q"].to_list()])) \
        .group_by("s1", "src").agg(pl.len().alias("n_other"))
x = pool.join(oc, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
s1f = pl.DataFrame({"s1": s1ids}, schema={"s1": pl.UInt32})
base = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oc, on=["s1", "src"], how="left") \
         .with_columns(pl.col("n_other").fill_null(0)).group_by("src", "n_other").agg(pl.len().alias("n_s1"))
t = x.group_by("src", "n_other", "cat").agg(pl.len().alias("n")).join(base, on=["src", "n_other"]).with_columns((pl.col("n") / pl.col("n_s1") * 1000).round(2).alias("per1k"))
print(t.pivot(on="cat", index=["src", "n_other"], values="per1k").fill_null(0).sort("src", "n_other"))
print("\ncounts and acceptance:", x.group_by("cat").agg(pl.len(), pl.col("acc").sum().alias("accepted")).sort("cat").rows())
print("B top swaps:", x.filter(pl.col("cat") == "B_exact_near_vocab").with_columns(pl.Series("pair", [f"{next(iter(nt[s] - nt[q]))}->{next(iter(nt[q] - nt[s]))}" for q, s in
      zip(x.filter(pl.col("cat") == "B_exact_near_vocab")["q"].to_list(), x.filter(pl.col("cat") == "B_exact_near_vocab")["s1"].to_list())])).group_by("pair").len().sort("len", descending=True).head(15).rows())
x.select("q", "s1", "cat", "acc", "src").write_parquet("/work/dive/fr_swap_subsets.parquet")
