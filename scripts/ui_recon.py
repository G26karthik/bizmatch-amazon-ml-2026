"""Reconstruct the production US/India decisions on test without AWS, and splice new US/India probabilities in.
approx test_best = uncertain best-S1 pairs (local_data/llm/unc_test.parquet, 0.02 < p < 0.98) + every accepted pair of the
base submission not in that file (p = 0.999). Re-running the production rule (efopt, gamma 1.5) on it must reproduce the
base submission's US/India rows (checked and reported). With --newp <parquet q, s1, pn> the uncertain probabilities are
replaced by pn (France rows are never changed) and the result is written to --out.

uv run --no-project --with-requirements requirements.txt python scripts/ui_recon.py --base output/v28q_fr15 [--newp X --out output/v28q_frXX]
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--newp", type=Path)
ap.add_argument("--out", type=Path)
a = ap.parse_args()
CFG = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet").select("entity_id", "country") for s in (1, 2, 3)]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(a.base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
ui_s1 = rec.filter(pl.col("country") != "France").select(pl.col("rid").alias("s1"))
acc_ui = acc.join(ui_s1, on="s1", how="semi")
unc = pl.read_parquet("local_data/llm/unc_test.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("prob").cast(pl.Float64)) \
        .join(ui_s1, on="s1", how="semi")
cert = acc_ui.join(unc.select("q"), on="q", how="anti").with_columns(pl.lit(0.999).alias("prob"))
best = pl.concat([unc, cert])
ch = _chosen(best, CFG).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
same = ch.join(acc_ui, on=["q", "s1"], how="inner").height
print(f"reconstruction: base US/India pairs {acc_ui.height}, reconstructed {ch.height}, identical {same} "
      f"(missing {acc_ui.height - same}, extra {ch.height - same})", flush=True)
if a.newp:
    npf = pl.read_parquet(a.newp).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("pn").cast(pl.Float64))
    b2 = pl.concat([unc.join(npf, on=["q", "s1"], how="left").with_columns(pl.coalesce("pn", "prob").alias("prob")).select("q", "s1", "prob"), cert])
    ch2 = _chosen(b2, CFG).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    # keep base decisions where the reconstruction already disagreed (only change what the new probabilities change)
    add = ch2.join(ch, on=["q", "s1"], how="anti")
    rem = ch.join(ch2, on=["q", "s1"], how="anti")
    new = pl.concat([acc.join(rem, on=["q", "s1"], how="anti"), add.join(acc.select("q"), on="q", how="anti")]).unique()
    assert new.select("q").is_duplicated().sum() == 0
    print(f"new probabilities: +{add.height} / -{rem.height} US/India pairs", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    m = new.join(eid.rename({"rid": "q", "entity_id": "e"}), on="q").join(eid.rename({"rid": "s1", "entity_id": "source1_entity_id"}), on="s1") \
           .sort("e").group_by("source1_entity_id").agg(pl.col("e").str.join(",").alias("matched_entity_ids"))
    o = g.select("source1_entity_id").with_row_index("ord").join(m, on="source1_entity_id", how="left").sort("ord").with_columns(pl.col("matched_entity_ids").fill_null(""))
    with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s, e in o.select("source1_entity_id", "matched_entity_ids").iter_rows():
            f.write(f"{s}\t{e}\n")
    (a.out / "candidate_pairs.tsv").write_bytes((a.base / "candidate_pairs.tsv").read_bytes())
    print("wrote", a.out)
