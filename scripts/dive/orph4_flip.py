"""Group table (validation true rate with / without hidden-parent orphans H, v27 decisions, test counts) and flip
simulations on half B: accept the rejected records of a group (their best S1) and measure the macro F0.5 change,
standard (all records) and test-like (H records removed; raw-orphan FPs x1 / x2).  Needs orph2 outputs.
"""
import json
from pathlib import Path
import numpy as np, polars as pl
from ber.pipeline import SEED, labels
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
OUT = Path("/work/dive/orph")
v = pl.read_parquet(OUT / "val_feats.parquet")
t = pl.read_parquet(OUT / "test_feats.parquet")
lab = labels(Path("/work/data"), Path("/work/base")).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
ntrue = lab.filter(pl.col("parent").is_not_null()).group_by(pl.col("parent").cast(pl.UInt32).alias("s1")).agg(pl.len().alias("nt"))
s1b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1")).join(ntrue, on="s1", how="left").fill_null(0)
NB, NT = s1b.height, 1_730_000
print("half-B S1s", NB, flush=True)


def grp(d, full_col):
    """group label; on validation the name frequency uses the pool (visible S1s) for ufreq and the full list for the
    test-like split of unique names."""
    hc = pl.when(pl.col("hrel").is_in(["h_eq"])).then(pl.lit("heq")).when(pl.col("hrel") == "h_diff").then(pl.lit("hdiff")) \
           .when(pl.col("hrel") == "empty").then(pl.lit("-")).otherwise(pl.lit("hmiss"))
    return d.with_columns(pl.concat_str([pl.when(pl.col("q_addr_empty")).then(pl.lit("EMPTY")).otherwise(pl.lit("addr")),
                                         pl.col("nrel"), hc, pl.col("ufreq"),
                                         pl.when(pl.col("decoy")).then(pl.lit("DECOY")).otherwise(pl.lit(""))], separator="|").alias("g"))


v = grp(v, "n_full").with_columns((pl.col("cls") == "H").alias("isH"))
t = grp(t, "n_full")
# sibling support: other records with the same normalised name whose best candidate is the same S1
v = v.with_columns((pl.len().over("s1", "q_name_n") - 1).alias("sib"))
t = t.with_columns((pl.len().over("s1", "q_name_n") - 1).alias("sib"))
vv = v.filter(pl.col("s1_val"))
for nm, cond in (("EMPTY exact", pl.col("q_addr_empty") & (pl.col("nrel") == "exact")),
                 ("EMPTY nonexact", pl.col("q_addr_empty") & (pl.col("nrel") != "exact")),
                 ("addr exact", ~pl.col("q_addr_empty") & (pl.col("nrel") == "exact")),
                 ("addr nonexact", ~pl.col("q_addr_empty") & (pl.col("nrel") != "exact"))):
    a1 = vv.filter(cond).group_by(pl.col("sib").clip(0, 2).alias("sib"), pl.col("ufreq")).agg(
        pl.len().alias("v_n"), (pl.col("cls") == "H").mean().round(3).alias("H_sh"), (pl.col("cls") == "R").mean().round(3).alias("R_sh"),
        pl.col("y").mean().round(3).alias("true_all"), pl.col("y").filter(pl.col("cls") != "H").mean().round(3).alias("true_noH"),
        pl.col("prob").mean().round(3).alias("v_mprob"), pl.col("acc").mean().round(3).alias("v_acc"),
        pl.col("y").filter(~pl.col("acc") & (pl.col("cls") != "H")).mean().round(3).alias("rej_true_noH"))
    a2 = t.filter(cond).group_by(pl.col("sib").clip(0, 2).alias("sib"), pl.col("ufreq")).agg(
        pl.len().alias("t_n"), pl.col("prob").mean().round(3).alias("t_mprob"), pl.col("acc").mean().round(3).alias("t_acc"),
        (~pl.col("acc")).sum().alias("t_rej"))
    print(nm, "by sibling support x S1-name frequency")
    print(a1.join(a2, on=["sib", "ufreq"], how="full", coalesce=True).sort("ufreq", "sib"))
gv = vv.group_by("g").agg(pl.len().alias("v_n"), pl.col("isH").mean().round(3).alias("H_sh"), (pl.col("cls") == "R").mean().round(3).alias("R_sh"),
                          pl.col("y").mean().round(3).alias("true_all"), pl.col("y").filter(~pl.col("isH")).mean().round(3).alias("true_noH"),
                          pl.col("prob").mean().round(3).alias("v_mprob"), pl.col("acc").mean().round(3).alias("v_acc"),
                          pl.col("y").filter(~pl.col("acc")).mean().round(3).alias("rej_true_all"),
                          pl.col("y").filter(~pl.col("acc") & ~pl.col("isH")).mean().round(3).alias("rej_true_noH"),
                          pl.col("prob").filter(~pl.col("acc") & ~pl.col("isH")).mean().round(3).alias("rej_mprob_noH"),
                          (~pl.col("acc") & ~pl.col("isH")).sum().alias("v_rej_noH"),
                          pl.col("y").filter(pl.col("acc") & ~pl.col("isH")).mean().round(3).alias("acc_prec_noH"))
gt = t.group_by("g").agg(pl.len().alias("t_n"), pl.col("prob").mean().round(3).alias("t_mprob"), pl.col("acc").mean().round(3).alias("t_acc"),
                         (~pl.col("acc")).sum().alias("t_rej"), (~pl.col("acc") & (pl.col("prob") >= 0.2)).sum().alias("t_rej_p.2"))
tab = gv.join(gt, on="g", how="full", coalesce=True).sort("t_rej", descending=True, nulls_last=True)
print(tab.head(60))
tab.write_parquet(OUT / "group_table.parquet")
# unique-in-pool empty exact split by full-list uniqueness (validation only)
x = vv.filter(pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("n_pool") == 1))
print("EMPTY|exact|u1 split by full-list frequency (val):")
print(x.group_by((pl.col("n_full") == 1).alias("unique_full")).agg(pl.len(), pl.col("isH").mean().round(3).alias("H_sh"),
      pl.col("y").mean().round(4).alias("true"), pl.col("y").filter(~pl.col("isH")).mean().round(4).alias("true_noH"),
      pl.col("prob").mean().round(3).alias("mprob"), pl.col("acc").mean().round(3).alias("acc"),
      pl.col("prob").filter(~pl.col("isH")).mean().round(3).alias("mprob_noH"), pl.col("acc").filter(~pl.col("isH")).mean().round(3).alias("acc_noH")))

# ---- flip simulations on half B
par = lab.filter(pl.col("src") != 1).select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32),
                                             pl.col("parent").is_null().alias("orph"))
Hq = v.filter(pl.col("isH")).select("q")


def macro(ch, k):
    c = ch.join(par, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("tp"))
    c = c.with_columns(pl.when(pl.col("tp")).then(0).when(pl.col("orph")).then(k).otherwise(1).alias("fp"))
    g = c.group_by("s1").agg(pl.col("tp").sum().cast(pl.Float64), pl.col("fp").sum().cast(pl.Float64))
    e = s1b.join(g, on="s1", how="left").fill_null(0)
    f = e.select(pl.when(pl.col("nt") == 0).then((pl.col("fp") == 0).cast(pl.Float64))
                 .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + pl.col("fp") + 0.25 * (pl.col("nt") - pl.col("tp")))).alias("f"))
    return float(f["f"].mean())


base = v.filter(pl.col("acc")).select("q", "s1")
base_tl = base.join(Hq, on="q", how="anti")
b0 = {"std_k1": macro(base, 1), "std_k3": macro(base, 3), "tl_k1": macro(base_tl, 1), "tl_k2": macro(base_tl, 2)}
print("baseline half B:", {k: round(x, 5) for k, x in b0.items()}, flush=True)
flips = {
    "EMPTY exact u1(pool)": pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("n_pool") == 1),
    "EMPTY exact u1 p>=.2": pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("n_pool") == 1) & (pl.col("prob") >= 0.2),
    "EMPTY exact u2": pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("n_pool") == 2),
    "EMPTY nonexact u0/u1 p>=.2": pl.col("q_addr_empty") & (pl.col("nrel") != "exact") & (pl.col("n_pool") <= 1) & (pl.col("prob") >= 0.2),
    "addr exact heq": ~pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("hrel") == "h_eq"),
    "addr exact hmiss p>=.2": ~pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & pl.col("hrel").is_in(["q_noh", "s_noh"]) & (pl.col("prob") >= 0.2),
    "addr exact hdiff p>=.2": ~pl.col("q_addr_empty") & (pl.col("nrel") == "exact") & (pl.col("hrel") == "h_diff") & (pl.col("prob") >= 0.2),
    "addr nondecoy heq p>=.2": ~pl.col("q_addr_empty") & (pl.col("nrel") != "exact") & ~pl.col("decoy") & (pl.col("hrel") == "h_eq") & (pl.col("prob") >= 0.2),
    "addr nondecoy heq p>=.4": ~pl.col("q_addr_empty") & (pl.col("nrel") != "exact") & ~pl.col("decoy") & (pl.col("hrel") == "h_eq") & (pl.col("prob") >= 0.4),
    "any p>=.4": pl.col("prob") >= 0.4,
    "any p>=.3": pl.col("prob") >= 0.3,
}
rows = []
for name, cond in flips.items():
    add = v.filter(~pl.col("acc") & cond).select("q", "s1")
    ch = pl.concat([base, add]).unique()
    ch_tl = ch.join(Hq, on="q", how="anti")
    addB = add.join(s1b.select("s1"), on="s1", how="semi")
    addB_noH = addB.join(Hq, on="q", how="anti")
    r = {"group": name, "valB_added": addB.height, "valB_added_noH": addB_noH.height,
         "added_true_rate_noH": round(float(addB_noH.join(par, on="q").select((pl.col("parent") == pl.col("s1")).fill_null(False).mean()).item() or 0), 3),
         "d_std_k1": round(macro(ch, 1) - b0["std_k1"], 6), "d_std_k3": round(macro(ch, 3) - b0["std_k3"], 6),
         "d_tl_k1": round(macro(ch_tl, 1) - b0["tl_k1"], 6), "d_tl_k2": round(macro(ch_tl, 2) - b0["tl_k2"], 6),
         "test_flips": t.filter(~pl.col("acc") & cond).height}
    # LB estimate: per flipped test-like half-B record effect scaled to test flips
    per = (r["d_tl_k2"] * NB / max(addB_noH.height, 1))
    r["LB_est_tl_k2"] = round(per * r["test_flips"] / NT, 6)
    rows.append(r)
    print(r, flush=True)
pl.DataFrame(rows).write_parquet(OUT / "flips.parquet")
print(pl.DataFrame(rows))
