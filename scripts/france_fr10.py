"""fr10: on top of fr8b, remove accepted France same-address one-word swaps between two different type words (in-word not
a generic copy word, not a near-spelling / prefix of the out-word), legal form kept and not lowercase (the other cases
were removed in fr8), when the S1 keeps at least one other accepted record. Decoy share of such swaps is estimated 0.42-
0.73 (sibling decoys; US same-address siblings keep the legal form 72%); dropping one from an S1 with other matches pays
off above ~0.24-0.27. S1s whose only match is the swap keep it (break-even 0.5).

python scripts/france_fr10.py --base_out /work/out_fr8b --out /work/out_fr10 [--keep_single]
"""
import argparse
import shutil
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

ap = argparse.ArgumentParser()
ap.add_argument("--base_out", type=Path, default=Path("/work/out_fr8b"))
ap.add_argument("--out", type=Path, default=Path("/work/out_fr10"))
ap.add_argument("--all", action="store_true", help="also remove when the swap is the S1's only match")
a = ap.parse_args()
GEN = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
sw = pl.read_parquet("/work/dive/fr_swaps_feat2.parquet").filter((pl.col("cat") == "same|TYPE") & (pl.col("legal_rel") == "same") & ~pl.col("lower"))
sw = sw.join(pl.read_parquet("/work/dive/fr_swaps.parquet").select("q", "s1", "w_in").unique(["q", "s1"]), on=["q", "s1", "w_in"], how="semi")
# swapped-out word
full = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name"]).with_row_index("rid")
import re
import unicodedata


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
nm = full["business_name"].to_list()


def abbrev(x, y):
    """x is an abbreviation of y: same first letter and x's letters appear in order in y (st/saint, cb/club, svc/service)."""
    if not x or not y or x[0] != y[0] or len(x) >= len(y):
        return False
    it = iter(y)
    return all(ch in it for ch in x)

wout, near = [], []
for q, s, wi in zip(sw["q"].to_list(), sw["s1"].to_list(), sw["w_in"].to_list()):
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEGAL
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEGAL
    wo = next(iter(B - A), "")
    wout.append(wo)
    near.append(fuzz.ratio(wi, wo) >= 70 or wi.startswith(wo[:4]) or wo.startswith(wi[:4]) or abbrev(wi, wo) or abbrev(wo, wi))
sw = sw.with_columns(pl.Series("w_out", wout), pl.Series("near", near)).filter(~pl.col("near") & ~pl.col("w_in").is_in(GEN))
cand = sw.select("q", "s1").unique().join(acc, on=["q", "s1"])
k = acc.group_by("s1").agg(pl.len().alias("k"))
r = cand.group_by("s1").agg(pl.len().alias("r"))
cand = cand.join(k, on="s1").join(r, on="s1")
single = cand.filter(pl.col("k") == pl.col("r"))
rm = cand if a.all else cand.filter(pl.col("k") > pl.col("r"))
print(f"candidates {cand.height} (S1 would go empty: {single.height}); removing {rm.height}")
print("top swaps (out -> in):", sw.join(rm.select("q", "s1"), on=["q", "s1"]).group_by("w_out", "w_in").len().sort("len", descending=True).head(15).rows())
new = acc.join(rm.select("q", "s1"), on=["q", "s1"], how="anti")
s1all = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
m = new.join(eid.rename({"rid": "q", "entity_id": "m"}), on="q").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids"))
o = s1all.join(m, on="s1", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).select("source1_entity_id", "matched_entity_ids")
o = o.join(g.select("source1_entity_id").with_row_index("ord"), on="source1_entity_id").sort("ord").drop("ord")
a.out.mkdir(parents=True, exist_ok=True)
with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s, mm in o.iter_rows():
        f.write(f"{s}\t{mm}\n")
shutil.copy(a.base_out / "candidate_pairs.tsv", a.out / "candidate_pairs.tsv")
rm.write_parquet(a.out / "fr10_removed.parquet")
fr = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
kk = fr.join(new.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
print("France k distribution after:", kk.group_by(pl.col("len").alias("k")).agg(pl.len().alias("n")).with_columns((pl.col("n") / kk.height).round(4).alias("p")).sort("k").head(6).rows(), "mean", kk["len"].mean())
print("wrote", a.out)
