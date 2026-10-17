"""E00 part 7: final summary tables (recoverable vs irreducible) and examples of the top recoverable bucket."""
from pathlib import Path

import polars as pl

OUT = Path("/work/agent_e00")
NB = 89043
e = pl.read_parquet(OUT / "err_records.parquet")
r2 = pl.read_parquet(OUT / "retr2.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "rank_base", "rank_nocap", "rank_bigram")
e = e.join(r2, on=["q", "s1"], how="left")
grp = (pl.when(pl.col("cat").str.starts_with("FP")).then(pl.lit("FP"))
         .when(pl.col("recov").is_in(["rec_parent_best_clear", "amb_parent_best_narrow"])).then(pl.lit("plausible: parent is unique best"))
         .otherwise(pl.lit("irreducible-looking: namesake / competitor as good")))
stage = (pl.when(pl.col("cat").is_in(["FN_not_retrieved", "FN_pruned_A<0.001", "FN_pruned_A_0.001-0.01"])).then(pl.lit("never scored"))
           .when(pl.col("cat").str.starts_with("FN")).then(pl.lit("scored, not chosen / wrong S1")).otherwise(pl.lit("false positive")))
e = e.with_columns(grp.alias("grp"), stage.alias("stage"))
cfg = pl.Config(tbl_rows=50, tbl_width_chars=220, fmt_str_lengths=60)
x = e.group_by("stage", "grp").agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05_loss"),
                                   ((pl.col("loss") * pl.col("acc").fill_null(0)).sum() / NB).alias("analogue_rec"))
with cfg:
    print("=== loss by stage x recoverability (analogue_rec only meaningful for 'never scored') ===")
    print(x.sort("stage", "grp"))
    print(e.group_by("grp").agg(pl.len(), (pl.col("loss").sum() / NB).alias("F05")).sort("F05", descending=True))

# FP split
with cfg:
    print(e.filter(pl.col("stage") == "false positive").group_by("cat", "q_addr_empty").agg(pl.len(), (pl.col("loss").sum() / NB).alias("F05")))

# scored-but-missed with a clear best parent: model probabilities
s = e.filter((pl.col("stage") == "scored, not chosen / wrong S1") & (pl.col("grp") == "plausible: parent is unique best"))
print("\nscored FN with unique-best parent:", s.height, "loss", round(s["loss"].sum() / NB, 6), "prob quartiles",
      [round(s["am_p"].quantile(q), 3) for q in (0.25, 0.5, 0.75)])

# top bucket: never retrieved, empty address; interventions
nr = e.filter((pl.col("cat") == "FN_not_retrieved") & pl.col("q_addr_empty"))
print(f"\nnever-retrieved empty-address: n={nr.height} loss={nr['loss'].sum() / NB:.6f} analogue_rec={(nr['loss'] * nr['acc'].fill_null(0)).sum() / NB:.6f}")
for v in ("rank_nocap", "rank_bigram"):
    for K in (8, 40):
        h = nr.filter(pl.col(v) < K)
        print(f"  {v} < {K}: n={h.height} loss={h['loss'].sum() / NB:.6f} analogue_rec={(h['loss'] * h['acc'].fill_null(0)).sum() / NB:.6f} "
              f"| of which parent unique best {h.filter(pl.col('grp').str.starts_with('plausible')).height}")
ex = nr.filter((pl.col("rank_nocap") < 40) & (pl.col("recov") == "rec_parent_best_clear")).sort("loss", descending=True)
print("\n=== examples: never retrieved, empty address, parent unique best by name, found by uncapped name retrieval (rank < 40) ===")
for r in ex.head(14).iter_rows(named=True):
    print(f"[{r['country']}|S{r['q_src']}|T={r['T']}|rank base={r['rank_base']} nocap={r['rank_nocap']} bigram={r['rank_bigram']}|"
          f"pool S1s with record name={r['q_name_cnt']}, with S1 name={r['s_name_cnt']}|name sim parent {r['p_sc']:.0f} vs best other {r['o_sc']:.0f}]")
    print(f"    record: {r['q_raw_name']!r} (no address)")
    print(f"    S1    : {r['s_raw_name']!r} @ {r['s_raw_addr']!r}")
ex2 = e.filter((pl.col("cat") == "FN_pruned_A_0.001-0.01") & (pl.col("recov") == "rec_parent_best_clear")).sort("loss", descending=True)
print("\n=== examples: Stage-A pruned (0.001 <= pA < 0.01), parent unique best ===")
for r in ex2.head(6).iter_rows(named=True):
    print(f"[{r['country']}|S{r['q_src']}|T={r['T']}|pA={r['pA']:.4f}] {r['q_raw_name']!r} @ {r['q_raw_addr']!r}  ->  {r['s_raw_name']!r} @ {r['s_raw_addr']!r}")
