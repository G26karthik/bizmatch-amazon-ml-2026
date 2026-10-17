"""agent_mt: inspect test residual picks and train residual-like OOF subsets with noise reweighting."""
import sys
import polars as pl
sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks  # noqa: E402
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
te = pl.read_parquet(f"{W}/test_scored.parquet")
tpk = picks(te, "p_bin")
cols = ["pr_dmean", "pre_dmean", "sub_dmean", "jf_dmean", "js_dmean", "cp_tot_dmean", "c_len_dmean", "addr_exact", "cp_tot", "n_cand"]
print("test p>=0.85 US/India pick feature quantiles")
x = tpk.filter((pl.col("p_bin") >= 0.85) & (pl.col("country") != "France"))
print(x.select(cols).describe())
print(x.select("country", "mname", "sname", "kind", "p_bin", "pr", "pre", "sub", "jf_dmean", "js_dmean", "cp_tot").sort("p_bin").head(40))
# all rows of a few groups
g = x.sort("p_bin")["m"].head(8).to_list()
print(te.filter(pl.col("m").is_in(g)).select("m", "mname", "sname", "p_bin", "pr", "pre", "jf", "js", "cp_tot", "addr_exact").sort("m"))
oof = pl.read_parquet(f"{W}/oof.parquet")
pk = picks(oof, "p_bin")
for nm, cond in [("pr_dmean<10 & pre_dmean<2 & sub_dmean<3", (pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3)),
                 ("pr_dmean<5 & pre_dmean<1 & sub_dmean<1", (pl.col("pr_dmean") < 5) & (pl.col("pre_dmean") < 1) & (pl.col("sub_dmean") < 1)),
                 ("pr<60 & pr_dmean<10", (pl.col("pr") < 60) & (pl.col("pr_dmean") < 10))]:
    s = pk.filter(cond)
    print(f"\nOOF residual-like [{nm}]: groups {s.height}, owner-in-cands {s['has_pos'].mean():.4f}, noise {1 - s['m_has_owner'].mean():.4f}")
    for t in (0.8, 0.85, 0.9, 0.95):
        z = s.filter(pl.col("p_bin") >= t)
        if z.height == 0:
            print(f"  p>={t}: n=0"); continue
        nz = z.filter(pl.col("m_has_owner") == 0); wr = z.filter((pl.col("m_has_owner") == 1) & (pl.col("label") == 0))
        for wn in (1, 10, 20):   # noise upweighted wn x (residual enrichment)
            prec = z["label"].sum() / (z["label"].sum() + wr.height + wn * nz.height)
            print(f"  p>={t}: n={z.height} prec={z['label'].mean():.4f} wrong_owner={wr.height} noise={nz.height}  prec@noise x{wn}={prec:.4f}")
    print(s.filter(pl.col("p_bin") >= 0.85).select(cols).mean())
