"""Splice new US/India decisions into an existing submission, keeping its France rows unchanged.
US/India pairs = production decision on <work>/test_best_<tag>.parquet (cfg from val_report_<tag>.json, as in
france_combo.py) or a given chosen-pairs parquet (q, s1 rids), + extra rule pairs (unassigned records only) + cap trim
(5 S2 / 6 S3 per S1, most probable kept). France rows (and any France edit parquet: --fr_remove / --fr_add, q s1 rids)
come from --base_out. Writes <out>/matching_results.tsv in the base row order and <out>/candidate_pairs.tsv = base
candidates + any new matches.

python scripts/splice_ui.py --base_out /work/out_fr12 --tag run28_7q --out /work/out_chk   (must reproduce the base)
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base_out", type=Path, required=True)
ap.add_argument("--work", type=Path, default=Path("/work/run28"))
ap.add_argument("--tag", default="run28_7q")
ap.add_argument("--chosen", type=Path, help="parquet q, s1 (rids) replacing the decision on test_best")
ap.add_argument("--extra_pairs", type=Path, default=Path("/work/dive/audit_recall/a10_initials_test.parquet"))
ap.add_argument("--fr_remove", type=Path, nargs="*", default=[])
ap.add_argument("--fr_add", type=Path, nargs="*", default=[])
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
fr_s1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
srcs = rec.select(pl.col("rid").alias("q"), "src")


def read_pairs(out):
    g = pl.read_csv(out / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    p = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
         .filter(pl.col("matched_entity_ids") != "") \
         .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
         .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
    return g.select("source1_entity_id"), p


order, base = read_pairs(a.base_out)
fr = base.join(fr_s1, on="s1", how="semi")
ui_old = base.join(fr_s1, on="s1", how="anti")
for f in a.fr_remove:
    r = pl.read_parquet(f).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).join(fr_s1, on="s1", how="semi")
    n0 = fr.height; fr = fr.join(r, on=["q", "s1"], how="anti"); print(f"France remove {f.name}: -{n0 - fr.height}")
for f in a.fr_add:
    r = pl.read_parquet(f).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).join(fr_s1, on="s1", how="semi") \
          .unique(subset="q", keep="none").join(fr.select("q"), on="q", how="anti")
    fr = pl.concat([fr, r]); print(f"France add {f.name}: +{r.height}")

best = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
if a.chosen:
    ui = pl.read_parquet(a.chosen).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
else:
    cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
    print("cfg", cfg)
    ui = _chosen(best, cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ui = ui.join(fr_s1, on="s1", how="anti")
if a.extra_pairs:
    ex = pl.read_parquet(a.extra_pairs).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).unique(subset="q", keep="none") \
           .join(fr_s1, on="s1", how="anti").join(ui.select("q"), on="q", how="anti")
    ui = pl.concat([ui, ex]); print(f"extra pairs: +{ex.height}")
x = ui.join(best.select("q", "s1", "prob"), on=["q", "s1"], how="left").with_columns(pl.col("prob").fill_null(0.5)).join(srcs, on="q") \
      .with_columns(pl.col("prob").rank("ordinal", descending=True).over("s1", "src").alias("_r"))
over = x.filter(((pl.col("src") == 2) & (pl.col("_r") > 5)) | ((pl.col("src") == 3) & (pl.col("_r") > 6))).select("q", "s1")
ui = ui.join(over, on=["q", "s1"], how="anti"); print(f"cap trim: -{over.height}")
d_add = ui.join(ui_old, on=["q", "s1"], how="anti").height
d_rem = ui_old.join(ui, on=["q", "s1"], how="anti").height
print(f"US/India pairs {ui_old.height} -> {ui.height}  (+{d_add} / -{d_rem})  France pairs {base.height - ui_old.height} -> {fr.height}")
cur = pl.concat([ui, fr])
assert cur.select("q").is_duplicated().sum() == 0, "a record was assigned twice"
x = cur.join(srcs, on="q").group_by("s1", "src").len()
assert x.filter(((pl.col("src") == 2) & (pl.col("len") > 5)) | ((pl.col("src") == 3) & (pl.col("len") > 6))).height == 0, "cap violated"
a.out.mkdir(parents=True, exist_ok=True)
m = cur.join(eid.rename({"rid": "q", "entity_id": "e"}), on="q").join(eid.rename({"rid": "s1", "entity_id": "source1_entity_id"}), on="s1") \
       .sort("e").group_by("source1_entity_id").agg(pl.col("e").str.join(",").alias("matched_entity_ids"))
o = order.with_row_index("ord").join(m, on="source1_entity_id", how="left").sort("ord").with_columns(pl.col("matched_entity_ids").fill_null(""))
with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s, e in o.select("source1_entity_id", "matched_entity_ids").iter_rows():
        f.write(f"{s}\t{e}\n")
c = (a.base_out / "candidate_pairs.tsv").read_bytes()
(a.out / "candidate_pairs.tsv").write_bytes(c)
print("wrote", a.out, o.height, "rows; run scripts/merge_candidates.py next")
