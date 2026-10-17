"""agent_mt: final picks. Test unassigned one-token records at 2-3-tenant addresses, model p >= THR, pick not tied on
address Jaccard (jf) or strong-name partial_ratio with another candidate, S2/S3 caps vs fr27 respected.
US/India -> local_data/work/agent_mt_adds.parquet; France (unvalidated) -> local_data/work/agent_mt_adds_france_unvalidated.parquet"""
import sys
import polars as pl
sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks  # noqa: E402
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
# guarded marginal bins on OOF residual-like
o = picks(pl.read_parquet(f"{W}/oof.parquet"), "p_bin").join(pl.read_parquet(f"{W}/oof_tie.parquet"), on="m")
rl = o.filter((pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3) & ~pl.col("tie_addr") & ~pl.col("tie_name") & (pl.col("p_bin") >= 0.7))
print("OOF residual-like, guarded, by bin:")
print(rl.with_columns(bin=pl.col("p_bin").cut([0.8, 0.85, 0.9, 0.95, 0.98], left_closed=True)).group_by("bin")
      .agg(pl.len(), prec=pl.col("label").mean().round(4), noise=(pl.col("m_has_owner") == 0).sum()).sort("bin"))
g_all = o.filter(~pl.col("tie_addr") & ~pl.col("tie_name") & (pl.col("p_bin") >= THR))
print(f"OOF all guarded p>={THR}: n={g_all.height} prec={g_all['label'].mean():.4f}")
te = picks(pl.read_parquet(f"{W}/test_scored.parquet"), "p_bin").join(pl.read_parquet(f"{W}/test_tie.parquet"), on="m")
sel = te.filter((pl.col("p_bin") >= THR) & ~pl.col("tie_addr") & ~pl.col("tie_name")).select("m", "s1", "country", "kind", pl.col("p_bin").alias("p"), "src")
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
ex = g.select(s1=pl.col("source1_entity_id"), m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
assert sel.join(ex, on="m").height == 0, "pick already assigned"
cnt = ex.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("s1", "src").len()
sel = sel.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0))
sel = sel.sort("p", descending=True).with_columns(k=pl.int_range(pl.len()).over("s1", "src") + 1)
sel = sel.with_columns(ok=(pl.col("len") + pl.col("k")) <= pl.when(pl.col("src") == "S2").then(5).otherwise(6))
print("cap drops:", sel.filter(~pl.col("ok")).height)
sel = sel.filter(pl.col("ok"))
assert sel["m"].n_unique() == sel.height
usin = sel.filter(pl.col("country") != "France").select("m", "s1", "country", "kind", "p").sort("country", "p")
fr = sel.filter(pl.col("country") == "France").select("m", "s1", "country", "kind", "p").sort("p")
print("US/India adds:"); print(usin.group_by("country", "kind").agg(pl.len(), pl.col("p").min().round(3)).sort("country", "kind"))
print("France (unvalidated):"); print(fr.group_by("kind").agg(pl.len(), pl.col("p").min().round(3)).sort("kind"))
usin.write_parquet("local_data/work/agent_mt_adds.parquet")
fr.write_parquet("local_data/work/agent_mt_adds_france_unvalidated.parquet")
print("wrote", usin.height, fr.height)
x = te.join(usin.select("m"), on="m")
pl.Config.set_fmt_str_lengths(40)
print(x.select("country", "mname", "sname", "p_bin", "jf_dmean", "pr", "addr_exact").sort("p_bin").head(25))
