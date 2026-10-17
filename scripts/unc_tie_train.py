"""Namesake-tie resolver for empty-address records (local).  Per-namesake LightGBM (legal-form resolver cues + weak
cues), pick = argmax per record, P(pick) calibrated by cross-fitted isotonic on the target population, expected-F0.5 add
rule (p^1.5 > (20k+4)/(25k+8)), caps, realized gain on train truth.

Pop A: no namesake in the record's candidate list (production never scored the tie); train stand-in = no namesake in
       pruned_pairs_train; records whose true parent was retrieved elsewhere weigh W_EX (production rejects ~54%).
Pop B: unassigned records with >=1 namesake in their candidate list (reported; production scores them).

uv run --no-project --with polars==1.44.2 --with numpy --with lightgbm --with scikit-learn python scripts/unc_tie_train.py [apply]
"""
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "6")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

sys.path.insert(0, "scripts")
from unc_train import decide, exp_gain, realized, thr  # noqa: E402

SCR = r"local_data/work"
N_S1_TEST = 1_732_544
W_EX = float(os.environ.get("W_EX", "0.55"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
ROW = ["legal_rel", "raw_eq", "rawnorm_eq", "full_eq", "seq_eq", "both_nolegal", "r_has_legal", "s_has_legal", "cap_full", "prior",
       "prior_cap", "post", "legal_mid", "s_legal_mid", "key_ntok", "key_len", "name_len", "s_name_len", "k_found", "k2", "k3",
       "s1_npp", "s_emp", "ns0", "n_same", "n_dropadd", "n_rawnorm", "n_seq", "n_full", "n_capfull", "k_rank", "k_minus_max",
       "k_mean", "npp_minus_max", "is_pick", "uniq_rawnorm", "uniq_seq", "n_rec_key", "n_rec_key_emp", "df_min", "df_lmean",
       "allcap", "src2", "nonascii"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=3)
ROUNDS = 500


def prep(R):
    return R.with_columns((pl.col("src") == 2).alias("src2"), pl.col("s_emp").fill_null(False))


def Xr(R):
    return R.select([pl.col(f).cast(pl.Float32) for f in ROW]).to_numpy()


def picks(R, score):
    R = R.with_columns(pl.Series("sc", score))
    R = R.with_columns(pl.col("sc").rank("ordinal", descending=True).over("q").alias("_r"),
                       pl.col("sc").sum().over("q").alias("sc_sum"))
    sec = R.filter(pl.col("_r") == 2).select("q", pl.col("sc").alias("sc2"))
    top = R.filter(pl.col("_r") == 1).join(sec, on="q", how="left").with_columns(pl.col("sc2").fill_null(0.0))
    return top.drop("_r")


def main():
    R = prep(pl.read_parquet(os.path.join(SCR, "tie_train.parquet")))
    ntrue = pl.read_parquet(os.path.join(SCR, "unc_train_ntrue.parquet"))
    R = R.with_columns(pl.when(pl.col("n_ret") == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("pop"),
                       pl.when(pl.col("par_ret") & ~pl.col("y").any().over("q")).then(W_EX).otherwise(1.0).alias("w"))
    # plain resolver precision by population x namesakes x cue (train truth)
    pk = R.filter(pl.col("is_pick"))
    print("RESOLVER (train truth) precision of the pick, by population / #namesakes / cue  [w = weight for parent-retrieved-elsewhere]")
    print(pk.group_by("pop", "country", "pick_cue", pl.col("ns0").clip(2, 5)).agg(pl.len(), ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(3).alias("prec"))
          .sort("pop", "country", "pick_cue", "ns0"))
    q = R.group_by("q").agg(pl.col("pop").first(), pl.col("country").first(), pl.col("y").any().alias("par_in"), pl.col("parent").first().is_null().alias("noncopy"))
    print("records: parent among namesakes / non-copy share:")
    print(q.group_by("pop", "country").agg(pl.len(), pl.col("par_in").mean().round(3), pl.col("noncopy").mean().round(3)).sort("pop", "country"))
    # ---- per-namesake model. Variant 'all': trained on every tie record (A+B); calibrated on Pop A picks only.
    grp = (pl.struct("country", "key").hash(5) % 5).alias("fold")
    R = R.with_columns(grp)
    fold = R["fold"].to_numpy(); X = Xr(R); y = R["y"].cast(pl.Int8).to_numpy(); wt = R["w"].to_numpy()
    res = {}
    for variant in ["all", "popA"]:
        m_ok = np.ones(len(y), bool) if variant == "all" else (R["pop"] == "A").to_numpy()
        sc = np.zeros(len(y)); models = []
        for f in range(5):
            tr = (fold != f) & m_ok; te = fold == f
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], weight=wt[tr]), ROUNDS)
            sc[te] = m.predict(X[te]); models.append(m)
        A = picks(R.filter(pl.col("pop") == "A"), sc[(R["pop"] == "A").to_numpy()])
        # cross-fitted isotonic on the top score (Pop A picks)
        ya = A["y"].cast(pl.Int8).to_numpy(); wa = A["w"].to_numpy(); s_top = A["sc"].to_numpy(); fa = A["fold"].to_numpy()
        pa = np.zeros(len(ya))
        for f in range(5):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_top[fa != f], ya[fa != f], sample_weight=wa[fa != f])
            pa[fa == f] = iso.predict(s_top[fa == f])
        iso_all = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_top, ya, sample_weight=wa)
        A = A.with_columns(pl.Series("p", pa)).join(ntrue, on="s", how="left").with_columns(pl.col("n_true").fill_null(0))
        ll = -np.average(ya * np.log(np.clip(pa, 1e-6, 1)) + (1 - ya) * np.log(np.clip(1 - pa, 1e-6, 1)), weights=wa)
        A = A.filter(~pl.col("cap_full"))
        D = decide(A, "p").filter(pl.col("add"))
        rg = realized(D) if D.height else 0.0
        wp = float((D["y"] * D["w"]).sum() / max(D["w"].sum(), 1e-9)) if D.height else float("nan")
        print(f"\n[{variant}] Pop A picks {A.height}: logloss {ll:.4f}, pick precision {np.average(ya, weights=wa):.3f}; adds {D.height}, weighted precision {wp:.3f}, "
              f"expected {sum(exp_gain(p, k) for p, k in zip(D['p'].to_list(), D['k_found'].to_list())):.1f}, realized {rg:.1f} S1-units")
        print(A.group_by(pl.col("p").cut([0.5, 0.63, 0.8, 0.9]).alias("pbin")).agg(pl.len(), pl.col("p").mean().round(3), ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(3).alias("prec")).sort("pbin"))
        if D.height:
            print(D.group_by("country", pl.col("ns0").clip(2, 4), pl.col("pick_cue").fill_null("none")).agg(pl.len(), ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(3).alias("prec"))
                  .sort("country", "ns0", "pick_cue"))
        res[variant] = (rg, models, iso_all)
    best = max(res, key=lambda v: res[v][0])
    print("chosen variant:", best, "realized", round(res[best][0], 1))
    imp = res[best][1][0].feature_importance("gain")
    print("importance:", sorted(zip(ROW, np.round(imp / imp.sum(), 3)), key=lambda t: -t[1])[:12])
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply(res[best][1], res[best][2], res[best][0])


def apply(models, iso, rg_train):
    T = prep(pl.read_parquet(os.path.join(SCR, "tie_test.parquet")))
    T = T.with_columns(pl.when(pl.col("n_ret") == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("pop"))
    q = T.group_by("q").agg(pl.col("country").first(), pl.col("pop").first(), pl.col("assigned").first(), pl.col("q_uprob").is_not_null().first().alias("in_unc"))
    print("\nTEST tie records (2..12 namesakes):")
    print(q.group_by("country", "pop", "assigned", "in_unc").len().sort("country", "pop", "assigned", "in_unc"))
    TA = T.filter((pl.col("pop") == "A") & ~pl.col("assigned"))
    sc = np.mean([m.predict(Xr(TA)) for m in models], axis=0)
    A = picks(TA, sc)
    A = A.with_columns(pl.Series("p", iso.predict(A["sc"].to_numpy()), dtype=pl.Float64))
    A = A.filter(~pl.col("cap_full"))
    print("test Pop A picks:", A.height, " p>0.63:", int((A["p"] > 0.63).sum()), " p>0.8:", int((A["p"] > 0.8).sum()))
    if rg_train <= 3.0:
        print(f"train realized gain {rg_train:.1f} <= 3 S1-units: no adds written (rule not supported by train truth)")
        D = A.head(0).with_columns(pl.lit(True).alias("add"))
    else:
        D = decide(A, "p").filter(pl.col("add"))
    D = D.with_columns(pl.Series("eg", [exp_gain(p, k) for p, k in zip(D["p"].to_list(), D["k_found"].to_list())], dtype=pl.Float64))
    print(D.group_by("country").agg(pl.len(), pl.col("p").mean().round(3), pl.col("eg").sum().round(1).alias("gain_S1")).sort("country"))
    tot = float(D["eg"].sum()) if D.height else 0.0
    print(f"TEST tie adds {D.height}; expected gain {tot:.1f} S1-units = {tot / N_S1_TEST:.6f} LB")
    D.select(pl.col("m_id").alias("m"), pl.col("s1_id").alias("s1"), "p", "country", "ns0", "pick_cue", "k_found", "eg", "business_name", "s_name") \
     .write_parquet(os.path.join(SCR, "tie_adds_full.parquet"))
    D.select(pl.col("m_id").alias("m"), pl.col("s1_id").alias("s1"), "p", "country").write_parquet(os.path.join(SCR, "tie_adds.parquet"))
    if D.height:
        print(D.sample(min(20, D.height), seed=1).select("country", "business_name", "s_name", "p", "ns0", "pick_cue", "k_found"))


if __name__ == "__main__":
    main()
