"""agent_mt: address-tie / name-tie guard. tie_addr = another candidate has the same full-token Jaccard (jf) as the pick;
tie_name = another candidate has the same partial_ratio (pr) >= 90 as the pick. Evaluated on OOF (all, residual-like) and
the production residual (val split); test counts."""
import sys
import polars as pl
sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks  # noqa: E402
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)


def tie(d):
    d = d.with_columns(jfmax=pl.col("jf").max().over("gid"), prmax=pl.col("pr").max().over("gid"))
    d = d.with_columns(n_jfmax=(pl.col("jf") == pl.col("jfmax")).sum().over("gid"), n_prtop=((pl.col("pr") >= 90) & (pl.col("pr") == pl.col("prmax"))).sum().over("gid"))
    return d


o = tie(pl.read_parquet(f"{W}/oof.parquet"))
pk = picks(o, "p_bin").with_columns(tie_addr=(pl.col("jf") == pl.col("jfmax")) & (pl.col("n_jfmax") >= 2), tie_name=(pl.col("n_prtop") >= 2) & (pl.col("pr") >= 90))
print(pk.filter(pl.col("p_bin") >= 0.85).group_by("tie_addr", "tie_name").agg(pl.len(), pl.col("label").mean().round(4)).sort("tie_addr", "tie_name"))
rl = pk.filter((pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3) & (pl.col("p_bin") >= 0.85))
print("residual-like p>=0.85:"); print(rl.group_by("tie_addr", "tie_name").agg(pl.len(), pl.col("label").mean().round(4)).sort("tie_addr", "tie_name"))
pr = pl.read_parquet(f"{W}/oof_prod.parquet") if False else None
te = tie(pl.read_parquet(f"{W}/test_scored.parquet"))
tpk = picks(te, "p_bin").with_columns(tie_addr=(pl.col("jf") == pl.col("jfmax")) & (pl.col("n_jfmax") >= 2), tie_name=(pl.col("n_prtop") >= 2) & (pl.col("pr") >= 90))
print("test p>=0.85:"); print(tpk.filter(pl.col("p_bin") >= 0.85).group_by("country", "tie_addr", "tie_name").len().sort("country", "tie_addr", "tie_name"))
pk.select("m", "tie_addr", "tie_name").write_parquet(f"{W}/oof_tie.parquet")
tpk.select("m", "tie_addr", "tie_name").write_parquet(f"{W}/test_tie.parquet")
