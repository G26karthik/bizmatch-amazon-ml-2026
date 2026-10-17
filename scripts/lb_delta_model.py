"""Expected leaderboard change of France-only edits as a function of the decoy share d of contested pairs (same-house
type-word swaps, exact or street-typo address). Per France S1, the union of pairs in the reference and each candidate
submission plus the S1's other candidate pairs (best-S1 records) are true independently: contested pairs with 1 - d,
other pairs with their blended probability (rule-added pairs without one: 0.9). Paired Monte-Carlo draws give
E[F_cand - F_ref]; the LB change is the France sum over all test S1s (full-test weights).

python scripts/lb_delta_model.py --ref /work/out_fr7 --cands /work/out_fr8b /work/out_fr10 /work/out_fr12 /work/out_fr11
"""
import argparse

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/work/out_fr7")
ap.add_argument("--cands", nargs="+", required=True)
ap.add_argument("--dv", nargs="+", type=float, default=[0.7, 0.85, 1.0], help="decoy share of vocabulary-word swaps")
ap.add_argument("--dt", nargs="+", type=float, default=[0.0, 0.1, 0.25], help="decoy share of typo (non-word) swaps")
ap.add_argument("--draws", type=int, default=64)
a = ap.parse_args()

rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "country", "src"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
N_ALL = rec.filter(pl.col("src") == 1).height
eid = rec.select("rid", "entity_id")
fr_s1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
            .filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1") \
            .join(fr_s1, on="s1", how="semi")


outs = [a.ref] + a.cands
acc = {o: accepted(o) for o in outs}
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)) \
       .join(fr_s1, on="s1", how="semi").select("q", "s1", "prob")
u = pl.concat([tb.select("q", "s1")] + [acc[o] for o in outs]).unique()
u = u.join(tb, on=["q", "s1"], how="left").with_columns(pl.col("prob").fill_null(0.9))
# contested = same-house type-word swaps + other one-word swaps; vocab = the new word is a French S1 vocabulary word
cv = pl.read_parquet("/work/lead/fr_contested_vocab.parquet").select("q", "s1", (pl.col("voc1") > 30).alias("vocab"))
ov = pl.read_parquet("/work/lead/fr_other_vocab_swaps.parquet").select("q", "s1", "vocab")
cont = pl.concat([cv, ov]).unique(["q", "s1"]).with_columns(pl.lit(True).alias("contested"))
u = u.join(cont, on=["q", "s1"], how="left").with_columns(pl.col("contested").fill_null(False), pl.col("vocab").fill_null(False))
for i, o in enumerate(outs):
    u = u.join(acc[o].with_columns(pl.lit(True).alias(f"in{i}")), on=["q", "s1"], how="left").with_columns(pl.col(f"in{i}").fill_null(False))
u = u.sort("s1")
print(f"France pairs in union {u.height}, contested {u['contested'].sum()}")
for i, o in enumerate(outs):
    print(f"  {o}: pairs {u[f'in{i}'].sum()}  contested accepted {u.filter(pl.col(f'in{i}') & pl.col('contested')).height}")
s1 = u["s1"].to_numpy()
starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]
n_fr = fr_s1.height
n_with = len(starts)
inm = np.stack([u[f"in{i}"].to_numpy() for i in range(len(outs))], 1).astype(np.float32)
nsel = np.add.reduceat(inm, starts, axis=0)                # S1 x outs
p0 = u["prob"].to_numpy().clip(0, 1); ct = u["contested"].to_numpy(); vc = u["vocab"].to_numpy()
rng = np.random.default_rng(0)
print(f"\nexpected LB change vs {a.ref} (full-test weights; France S1s with any pair {n_with} of {n_fr})")
print("dv   dt    " + "  ".join(f"{o.split('_')[-1]:>9}" for o in a.cands) + "     E[F_FR ref]")
for dv, dt in [(v, t) for v in a.dv for t in a.dt]:
    p = np.where(ct & vc, 1 - dv, np.where(ct, 1 - dt, p0))
    acc_f = np.zeros(len(outs))
    for _ in range(a.draws):
        y = (rng.random(len(p)) < p).astype(np.float32)
        T = np.add.reduceat(y, starts)
        tp = np.add.reduceat(inm * y[:, None], starts, axis=0)
        den = 0.25 * T[:, None] + nsel
        f = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-9), 1.0)
        acc_f += f.sum(0)
    ef = acc_f / a.draws + (n_fr - n_with)                 # S1s with no pairs at all: empty = empty -> 1
    dl = (ef[1:] - ef[0]) / N_ALL
    print(f"{dv:.2f} {dt:.2f}  " + "  ".join(f"{x:+9.5f}" for x in dl) + f"     {ef[0] / n_fr:.4f}")
