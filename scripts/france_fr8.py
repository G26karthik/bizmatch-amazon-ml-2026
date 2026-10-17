"""fr8 France edits on top of a submission (default fr7):
  REMOVE  accepted France same-address one-word swaps to a type word (not a generic copy word) whose record carries an
          extra noise operation: legal form changed/dropped/added, or an all-lowercase name. France copies carry one
          operation (generic-swap copies keep the legal form 97.6%, lowercase 0.1%); decoys keep the legal form 51%,
          lowercase 4.6%; decoy share among same-address type swaps ~0.4 -> these records are ~94% decoys.
  ADD     (a) unaccepted same-address generic-word swaps (fils, cie, associes, services, ...) with the legal form kept,
          not lowercase, assigned to their S1 (US analog: 99.97% true);
          (b) unaccepted empty-address records whose name key equals exactly one France S1 name key, when the model's best
          S1 for the record is that S1 with prob >= --min_prob (validation analog: 69-87% true outside hidden-S1 orphans).
  Caps (<= 5 S2, <= 6 S3 per S1) are respected; additions never go to an S1 at a cap.

python scripts/france_fr8.py --base_out /work/out_fr7 --out /work/out_fr8 [--no_remove] [--no_add_gen] [--no_add_empty]
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/code/src")
from ber.stage2 import _write_outputs  # noqa: E402,F401  (kept for parity with other France scripts)

ap = argparse.ArgumentParser()
ap.add_argument("--base_out", type=Path, default=Path("/work/out_fr7"))
ap.add_argument("--out", type=Path, default=Path("/work/out_fr8"))
ap.add_argument("--tag", default="run28_7q")
ap.add_argument("--min_prob", type=float, default=0.2)
ap.add_argument("--no_remove", action="store_true")
ap.add_argument("--no_add_gen", action="store_true")
ap.add_argument("--no_add_empty", action="store_true")
a = ap.parse_args()
GEN = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", s)) - LEGAL))


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_empty"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
fr_s1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
sw = pl.read_parquet("/work/dive/fr_swaps_feat2.parquet")  # France one-swap records on the S1 street with legal/format features
n0 = acc.height
removed = pl.DataFrame(schema={"q": pl.UInt32, "s1": pl.UInt32})
if not a.no_remove:
    removed = sw.filter((pl.col("cat") == "same|TYPE") & ((pl.col("legal_rel") != "same") | pl.col("lower"))).select("q", "s1").unique() \
                .join(acc, on=["q", "s1"])
    acc = acc.join(removed, on=["q", "s1"], how="anti")
assigned = acc.select("q")
adds = []
if not a.no_add_gen:
    ag = sw.filter((pl.col("cat") == "same|GEN") & (pl.col("legal_rel") == "same") & ~pl.col("lower")).select("q", "s1").unique() \
           .join(assigned, on="q", how="anti")
    ag = ag.unique("q", keep="none")  # a record that is a generic swap of two S1s is ambiguous
    adds.append(ag.with_columns(pl.lit("gen_swap").alias("why")))
if not a.no_add_empty:
    rk = rec.with_columns(pl.Series("nk", [nkey(n) for n in rec["business_name"].to_list()]))
    s1k = rk.filter((pl.col("country") == "France") & (pl.col("src") == 1) & (pl.col("nk") != "")).group_by("nk").agg(pl.len().alias("n"), pl.col("rid").first().alias("s1"))
    s1k = s1k.filter(pl.col("n") == 1).select("nk", "s1")
    qk = rk.filter((pl.col("country") == "France") & (pl.col("src") != 1) & pl.col("addr_empty").fill_null(False) & (pl.col("nk") != "")) \
           .select(pl.col("rid").alias("q"), "nk")
    ae = qk.join(s1k, on="nk").select("q", "s1").join(assigned, on="q", how="anti")
    tb = pl.read_parquet(f"/work/run28/test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    ae = ae.join(tb, on=["q", "s1"]).filter(pl.col("prob") >= a.min_prob).select("q", "s1")
    adds.append(ae.with_columns(pl.lit("empty_unique").alias("why")))
add = pl.concat(adds).unique("q", keep="first") if adds else pl.DataFrame(schema={"q": pl.UInt32, "s1": pl.UInt32, "why": pl.String})
# caps
src = rec.select(pl.col("rid").alias("q"), "src")
cur = acc.join(src, on="q").group_by("s1", "src").len()
cap = {2: 5, 3: 6}
add = add.join(src, on="q").join(cur, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0))
add = add.with_columns(pl.int_range(pl.len()).over("s1", "src").alias("rank"))
add = add.filter(pl.col("len") + pl.col("rank") + 1 <= pl.col("src").replace_strict(cap, default=6)).select("q", "s1", "why")
print(f"base pairs {n0}; removed {removed.height}; added {add.height} {add.group_by('why').len().rows()}")
empties_before = fr_s1.join(acc.select("s1").unique(), on="s1", how="anti").height
new = pl.concat([acc, add.select("q", "s1")])
empties_after = fr_s1.join(new.select("s1").unique(), on="s1", how="anti").height
print(f"France empty S1s after removal {empties_before}, after additions {empties_after} (of {fr_s1.height})")
# write submission: all S1 rows, comma-joined matches
s1all = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
m = new.join(eid.rename({"rid": "q", "entity_id": "m"}), on="q").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids"))
o = s1all.join(m, on="s1", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).select("source1_entity_id", "matched_entity_ids")
o = o.join(g.select("source1_entity_id").with_row_index("ord"), on="source1_entity_id").sort("ord").drop("ord")
a.out.mkdir(parents=True, exist_ok=True)
with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s, mm in o.iter_rows():
        f.write(f"{s}\t{mm}\n")
removed.write_parquet(a.out / "fr8_removed.parquet"); add.write_parquet(a.out / "fr8_added.parquet")
cp = a.base_out / "candidate_pairs.tsv"
if cp.exists():
    import shutil
    shutil.copy(cp, a.out / "candidate_pairs.tsv")
print("wrote", a.out)
