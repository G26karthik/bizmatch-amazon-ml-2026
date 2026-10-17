"""fr13 France edit lists (vocabulary rule, from fr13_vocab_split.py): a contested same-house type-word swap is a copy when
the new word is not a French S1 vocabulary word (voc1 <= 30: typo of the S1 word; bracket share 6.0%), a sibling decoy when
it is one (voc1 > 30; bracket share 0.8%, like offset decoys).
  restore: removed in fr12 (accepted in fr7), voc1 <= 30, legal form kept, not lowercase (one name operation), record free
  remove : kept in fr12, voc1 > 30
Also audits every other accepted fr13 France pair whose name differs from the S1 by one swap to a vocabulary word.

python scripts/fr13_build_lists.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
LEG = LEGF | {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
GEN = {"fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie", "service"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def abbrev(x, y):
    if not x or not y or x[0] != y[0] or len(x) >= len(y):
        return False
    it = iter(y)
    return all(ch in it for ch in x)


x = pl.read_parquet("/work/lead/fr_contested_vocab.parquet")
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); hs = rec["house"].to_list(); ad = rec["addr_n"].to_list()
eid = rec.select("rid", "entity_id")
raw = lambda i: set(re.findall(r"[a-z0-9]+", norm(nm[i])))  # noqa: E731
x = x.with_columns(pl.Series("legal_same", [(raw(q) & LEGF) == (raw(s) & LEGF) for q, s in zip(x["q"].to_list(), x["s1"].to_list())]),
                   pl.Series("lower", [nm[q] == nm[q].lower() for q in x["q"].to_list()]))


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


a12 = accepted("/work/out_fr12")
rem_all = x.filter((pl.col("status") == "removed") & (pl.col("voc1") <= 30))
print("removed non-vocab swaps", rem_all.height, "by (legal_same, lower):", rem_all.group_by("legal_same", "lower").agg(pl.len(), pl.col("bracket").mean().round(4)).rows())
restore = rem_all.filter(pl.col("legal_same") & ~pl.col("lower")).join(a12.select("q"), on="q", how="anti")
remove = x.filter((pl.col("status") == "kept") & (pl.col("voc1") > 30))
print(f"restore {restore.height} (bracket {restore['bracket'].mean():.4f});  remove {remove.height} (bracket {remove['bracket'].mean():.4f})")
restore.select("q", "s1", "prob").write_parquet("/work/lead/fr13_restore.parquet")
remove.select("q", "s1", "prob").write_parquet("/work/lead/fr13_remove.parquet")

# ---- audit: all accepted fr12 France pairs with a one-word swap to a vocabulary word, outside the contested set
fr = rec.filter(pl.col("country") == "France")
voc1 = Counter(w for n in fr.filter(pl.col("src") == 1)["business_name"].to_list() for w in set(re.findall(r"[a-z0-9]+", norm(n))))
fr_s1 = fr.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"))
acc = a12.join(fr_s1, on="s1", how="semi").join(x.select("q", "s1"), on=["q", "s1"], how="anti")
rows = []
for q, s in zip(acc["q"].to_list(), acc["s1"].to_list()):
    A, B = raw(q) - LEG, raw(s) - LEG
    if not (len(A - B) == 1 and len(B - A) == 1 and A & B):
        continue
    wi, wo = next(iter(A - B)), next(iter(B - A))
    if wi in GEN or fuzz.ratio(wi, wo) >= 70 or abbrev(wi, wo) or abbrev(wo, wi) or wi.startswith(wo[:4]) or wo.startswith(wi[:4]):
        continue
    hq, hs1 = hs[q], hs[s]
    arel = "no_addr" if not ad[q] or not ad[s] else ("same_house" if hq and hq == hs1 else ("no_house" if not hq or not hs1 else "diff_house"))
    br = bool(re.search(r"[\[\(]\s*" + re.escape(wi), norm(nm[q])))
    rows.append((q, s, wi, wo, voc1.get(wi, 0), arel, br, (raw(q) & LEGF) == (raw(s) & LEGF), nm[q] == nm[q].lower()))
y = pl.DataFrame(rows, schema=["q", "s1", "new", "old", "voc1", "arel", "bracket", "legal_same", "lower"], orient="row", infer_schema_length=None)
y = y.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), (pl.col("voc1") > 30).alias("vocab"))
print("\n=== other accepted France one-word swaps (not generic, not near-spelling), by address relation x vocab")
print(y.group_by("arel", "vocab").agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk"),
      pl.col("legal_same").mean().round(3).alias("leg"), pl.col("lower").mean().round(3).alias("low"))
      .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).sort("arel", "vocab"))
for arel in y["arel"].unique().to_list():
    z = y.filter((pl.col("arel") == arel) & pl.col("vocab"))
    if z.height:
        print(f"\nexamples {arel} vocab swaps")
        for q, s in z.select("q", "s1").head(6).iter_rows():
            print(f"   {nm[s]!r:42} | {nm[q]!r:42} | {ad[s]} | {ad[q]}")
y.write_parquet("/work/lead/fr_other_vocab_swaps.parquet")
