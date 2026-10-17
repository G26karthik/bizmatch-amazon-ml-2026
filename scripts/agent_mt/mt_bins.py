"""agent_mt: marginal precision by p-bin (residual-like OOF subset, noise upweighted) + test pick counts per bin."""
import sys
import polars as pl
sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks  # noqa: E402
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
pk = picks(pl.read_parquet(f"{W}/oof.parquet"), "p_bin")
tpk = picks(pl.read_parquet(f"{W}/test_scored.parquet"), "p_bin")
edges = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]
def binit(d):
    return d.with_columns(bin=pl.col("p_bin").cut(edges[1:-1], left_closed=True))
for nm, cond in [("ALL", pl.lit(True)), ("residual-like pr_dmean<10&pre_dmean<2&sub_dmean<3", (pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3)),
                 ("residual-like & jf_dmean<0.2", (pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3) & (pl.col("jf_dmean") < 0.2))]:
    s = binit(pk.filter(cond & (pl.col("p_bin") >= 0.5)))
    print(f"\n{nm}")
    print(s.group_by("bin").agg(pl.len(), right=pl.col("label").sum(), noise=(pl.col("m_has_owner") == 0).sum(),
                                wrong_owner=((pl.col("m_has_owner") == 1) & (pl.col("label") == 0)).sum())
          .with_columns(prec=pl.col("right") / pl.len(), prec_n14=pl.col("right") / (pl.col("right") + pl.col("wrong_owner") + 14 * pl.col("noise")),
                        prec_n14_wo3=pl.col("right") / (pl.col("right") + 3 * pl.col("wrong_owner") + 14 * pl.col("noise"))).sort("bin"))
print("\nTEST picks per bin (US/India), jf_dmean mean:")
print(binit(tpk.filter((pl.col("country") != "France") & (pl.col("p_bin") >= 0.5))).group_by("country", "bin").agg(pl.len(), pl.col("jf_dmean").mean().round(3), pl.col("js_dmean").mean().round(3)).sort("country", "bin"))
print(binit(tpk.filter((pl.col("country") == "France") & (pl.col("p_bin") >= 0.5))).group_by("kind", "bin").agg(pl.len(), pl.col("jf_dmean").mean().round(3)).sort("kind", "bin"))
