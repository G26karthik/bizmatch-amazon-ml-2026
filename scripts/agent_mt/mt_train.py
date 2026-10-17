"""agent_mt: LightGBM on multi-tenant one-token rows. Binary per-row model with group-relative features, plus a
group-softmax (rank_xendcg) variant; split by crc32(akey) % 2 (train on 0, eval on 1, then swap for OOF).
Pick = argmax within group; report top-1 accuracy and precision/coverage at thresholds. Apply to test rows.

uv run --no-project --with polars==1.44.2 --with lightgbm --with numpy python scripts/agent_mt/mt_train.py
"""
import os
import sys
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
W = "local_data/work/agent_mt"
BASE = ["n_cand", "tok_len", "tok_alpha", "tok_digit", "tok_dot", "js", "jf", "cp_same", "cp_oth", "cp_tot", "cp_same_room",
        "al_same", "al_oth", "compet", "addr_exact", "n_q_addr", "initials", "ini_prefix", "pr", "prw", "rat", "handle", "sub",
        "sub_frac", "pre", "first", "legal", "c_ntok", "c_nws", "c_len", "is_s3", "ctry_us"]
REL = ["pr", "prw", "rat", "sub", "pre", "initials", "handle", "cp_tot", "cp_same", "js", "jf", "c_len"]


def prep(d):
    d = d.with_columns(is_s3=(pl.col("src") == "S3").cast(pl.Int8), ctry_us=(pl.col("country") == "US").cast(pl.Int8),
                       gid=pl.col("m"))
    ex = []
    for c in REL:
        ex += [(pl.col(c) - pl.col(c).max().over("gid")).alias(f"{c}_dmax"),
               (pl.col(c) - (pl.col(c).sum().over("gid") - pl.col(c)) / (pl.col("n_cand") - 1)).alias(f"{c}_dmean"),
               pl.col(c).rank("min", descending=True).over("gid").alias(f"{c}_rk")]
    d = d.with_columns(ex).sort("gid", "s1")
    return d


def feats():
    return BASE + [f"{c}_{s}" for c in REL for s in ("dmax", "dmean", "rk")]


PAR = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)


def fit_bin(tr, F):
    return lgb.train({**PAR, "objective": "binary"}, lgb.Dataset(tr.select(F).to_numpy(), tr["label"].to_numpy()), num_boost_round=400)


def fit_rank(tr, F):
    gs = tr.group_by("gid", maintain_order=True).len()["len"].to_numpy()
    return lgb.train({**PAR, "objective": "rank_xendcg"}, lgb.Dataset(tr.select(F).to_numpy(), tr["label"].to_numpy(), group=gs),
                     num_boost_round=400)


def picks(d, pcol):
    return (d.sort(pcol, descending=True).group_by("gid", maintain_order=True).first())


def report(pk, pcol, title):
    print(f"\n== {title}: groups {pk.height}, groups-with-owner-in-cands {pk['has_pos'].mean():.4f}, top1 acc {pk['label'].mean():.4f}")
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        s = pk.filter(pl.col(pcol) >= t)
        print(f"  p>={t:.2f}: n={s.height:6d} cov={s.height / max(1, pk.height):.4f} prec={s['label'].mean() if s.height else float('nan'):.4f}")


def main():
    tr = prep(pl.read_parquet(f"{W}/rows_train.parquet"))
    tr = tr.with_columns(fold=pl.col("akey").map_elements(lambda s: zlib.crc32(s.encode()) % 2, return_dtype=pl.Int64),
                         has_pos=pl.col("label").max().over("gid"))
    print(tr.group_by("country", "n_cand").agg(groups=pl.col("gid").n_unique(), pos=pl.col("label").sum(),
                                                owner=pl.col("m_has_owner").mean()).sort("country", "n_cand"))
    F = feats()
    parts = []
    for k in (0, 1):
        a, b = tr.filter(pl.col("fold") != k), tr.filter(pl.col("fold") == k)
        mb, mr = fit_bin(a, F), fit_rank(a, F)
        Xb = b.select(F).to_numpy()
        b = b.with_columns(p_bin=pl.Series(mb.predict(Xb)), s_rank=pl.Series(mr.predict(Xb)))
        parts.append(b)
    oof = pl.concat(parts)
    # group softmax of rank score; calibrate "any owner" via binary max; combined p = softmax * P(owner in cands)
    oof = oof.with_columns(sm=(pl.col("s_rank").exp() / pl.col("s_rank").exp().sum().over("gid")),
                           p_bin_norm=pl.col("p_bin") / pl.col("p_bin").sum().over("gid").clip(lower_bound=1.0))
    oof.write_parquet(f"{W}/oof.parquet")
    pk = picks(oof, "p_bin")
    report(pk, "p_bin", "OOF binary (both folds)")
    report(picks(oof.filter(pl.col("fold") == 1), "p_bin"), "p_bin", "held-out fold 1 binary (trained on fold 0)")
    report(picks(oof, "p_bin_norm"), "p_bin_norm", "OOF binary group-normalised")
    pks = picks(oof, "sm")
    report(pks, "sm", "OOF rank_xendcg group softmax (no 'none' option)")
    for c in ("US", "India"):
        report(pk.filter(pl.col("country") == c), "p_bin", f"OOF binary {c}")
    print("\nOOF binary picks p>=0.80 by kind/country/n_cand:")
    print(pk.filter(pl.col("p_bin") >= 0.8).group_by("country", "kind", "n_cand").agg(pl.len(), pl.col("label").mean().round(4)).sort("country", "kind", "n_cand"))
    print("\nall picks by kind (top1 acc, mean p):")
    print(pk.group_by("country", "kind").agg(pl.len(), pl.col("label").mean().round(4), pl.col("p_bin").mean().round(4)).sort("country", "kind"))
    # calibration bins
    print("\ncalibration of top pick p_bin:")
    print(pk.with_columns(bin=(pl.col("p_bin") * 10).floor() / 10).group_by("bin").agg(pl.len(), pl.col("label").mean().round(4)).sort("bin"))
    # final model on all train -> test
    mb = fit_bin(tr, F)
    imp = sorted(zip(F, mb.feature_importance("gain")), key=lambda x: -x[1])[:15]
    print("\ntop gain:", [(f, round(g)) for f, g in imp])
    mb.save_model(f"{W}/mt_bin.txt")
    if not os.path.exists(f"{W}/rows_test.parquet") or "notest" in sys.argv:
        return
    te = prep(pl.read_parquet(f"{W}/rows_test.parquet"))
    te = te.with_columns(p_bin=pl.Series(mb.predict(te.select(F).to_numpy())))
    te.write_parquet(f"{W}/test_scored.parquet")
    tpk = picks(te, "p_bin")
    print("\nTEST groups by country/kind and picks at thresholds:")
    print(tpk.group_by("country").agg(pl.len(), (pl.col("p_bin") >= 0.8).sum().alias("p80"), (pl.col("p_bin") >= 0.85).sum().alias("p85"),
                                      (pl.col("p_bin") >= 0.9).sum().alias("p90")).sort("country"))
    print(tpk.filter(pl.col("p_bin") >= 0.8).group_by("country", "kind").len().sort("country", "kind"))


if __name__ == "__main__":
    main()
