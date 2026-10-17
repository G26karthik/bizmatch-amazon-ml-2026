"""Train/calibrate P(true) for (empty-address record, unique-best uncapped name S1) pairs on train truth and
evaluate the expected-F0.5 add rule; then (mode apply) score the test pairs and write the add list.

Train population (analog of test 'not in any fr15 candidate list'): candidate S1 NOT in the record's retrieved
(pruned) pairs, and the record's true parent not retrieved either (such records would mostly be assigned to it).

uv run --no-project --with polars==1.44.2 --with numpy --with lightgbm python scripts/unc_train.py
"""
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "6")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SCR = r"local_data/work"
N_S1_TEST = 1_732_544
GAMMA = 1.5
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)

FEATS = ["tier", "is_ed1", "is_skel", "raw_eq", "ci_eq", "full_eq", "seq_eq", "legal_rel", "r_has_legal", "s_has_legal",
         "nonascii", "s_nonascii", "key_ntok", "key_len", "df_min", "df_max", "df_lmean", "allcap", "capfrac", "n_near",
         "n_skel", "n_rec_key", "n_rec_key_emp", "k_found", "k2", "k3", "k_samekey", "s1_npp", "src2", "s_emp",
         "name_len", "s_name_len", "n_super1", "n_minus1", "n_swap1", "legal_mid", "s_legal_mid", "is_super", "is_minus", "is_swap"]


def feats(C: pl.DataFrame) -> pl.DataFrame:
    rel = []
    for a, b in zip(C["legal"].to_list(), C["s_legal"].to_list()):
        ra, rb = set(filter(None, (a or "").split(","))), set(filter(None, (b or "").split(",")))
        rel.append(0 if not ra and not rb else 1 if ra == rb else 2 if ra < rb else 3 if rb < ra else 4)
    return C.with_columns(
        pl.Series("legal_rel", rel, dtype=pl.Int8),
        (pl.col("business_name") == pl.col("s_name")).alias("raw_eq"),
        (pl.col("business_name").str.to_lowercase().str.strip_chars() == pl.col("s_name").str.to_lowercase().str.strip_chars()).alias("ci_eq"),
        (pl.col("full") == pl.col("s_full")).alias("full_eq"),
        (pl.col("seq") == pl.col("s_seq")).alias("seq_eq"),
        (pl.col("legal") != "").alias("r_has_legal"), (pl.col("s_legal") != "").alias("s_has_legal"),
        pl.col("s_key").str.split(" ").list.len().alias("key_ntok"), pl.col("s_key").str.len_chars().alias("key_len"),
        (pl.col("src") == 2).alias("src2"),
        pl.col("business_name").str.len_chars().alias("name_len"), pl.col("s_name").str.len_chars().alias("s_name_len"),
    )


def X(C):
    return C.select([pl.col(f).cast(pl.Float32) for f in FEATS]).to_numpy()


def thr(k):
    return (20 * k + 4) / (25 * k + 8)


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=7)
ROUNDS = 600


def decide(C: pl.DataFrame, pcol: str) -> pl.DataFrame:
    """Greedy expected-F0.5 adds per S1 in descending p, respecting caps (S2<=5, S3<=6 incl. existing)."""
    C = C.sort(pcol, descending=True)
    k, k2, k3, keep = {}, {}, {}, []
    for s, p, src, kf, a2, a3 in zip(C["s"].to_list(), C[pcol].to_list(), C["src"].to_list(), C["k_found"].to_list(),
                                     C["k2"].to_list(), C["k3"].to_list()):
        kk = k.get(s, kf); c2 = k2.get(s, a2); c3 = k3.get(s, a3)
        ok = p ** GAMMA > thr(kk) and ((src == 2 and c2 < 5) or (src == 3 and c3 < 6))
        keep.append(ok)
        if ok:
            k[s] = kk + 1; k2[s] = c2 + (src == 2); k3[s] = c3 + (src == 3)
    return C.with_columns(pl.Series("add", keep))


def exp_gain(p, k):
    return p / (5 * k + 1) - (1 - p) * 4 / (5 * k + 4) if k > 0 else 2 * p - 1


def realized(A: pl.DataFrame, by=None) -> float:
    """train truth: realized F0.5 change per S1 (found copies assumed correct; f = true copies not found)."""
    if by is not None:
        out = {}
        for key, sub in A.group_by(by):
            out[key] = round(realized(sub), 1)
        return out
    g = 0.0
    for s, grp in A.group_by("s"):
        k = grp["k_found"][0]; n = grp["n_true"][0]
        tp_add = int(grp["y"].sum()); fp_add = float(grp.filter(~pl.col("y"))["w"].sum())
        f0 = n - k
        before = 1.0 if (k == 0 and f0 == 0) else 5 * k / (5 * k + f0)
        tp = k + tp_add; fn = f0 - tp_add; fp = fp_add
        after = 1.0 if (tp == 0 and fn == 0 and fp == 0) else (5 * tp / (5 * tp + fn + 4 * fp) if tp > 0 else 0.0)
        g += after - before
    return g


def main():
    C = feats(pl.read_parquet(os.path.join(SCR, "unc_train.parquet")))
    if os.environ.get("TESTLIKE") == "1":   # test has no copies of hidden S1s: drop those records from training
        rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet", columns=["entity_id"]).with_columns(pl.lit(s).alias("src"))
                         for s in (1, 2, 3)]).with_row_index("q").with_columns(pl.col("q").cast(pl.UInt32))
        hid = rec.filter((pl.col("src") == 1) & ((pl.col("entity_id").hash(42 + 77) % 100) < 19)).select(pl.col("entity_id").alias("s1"))
        tp = pl.read_parquet("local_data/train_pairs.parquet").join(hid, on="s1").select(pl.col("m").alias("entity_id"))
        H = rec.join(tp, on="entity_id").select("q")
        n0 = C.height
        C = C.join(H, on="q", how="anti")
        print(f"TESTLIKE: removed {n0 - C.height} copies of hidden S1s from {n0} train rows")
    ntrue = pl.read_parquet(os.path.join(SCR, "unc_train_ntrue.parquet")) if os.path.exists(os.path.join(SCR, "unc_train_ntrue.parquet")) else None
    print("train unique-best candidates:", C.height)
    print(C.group_by("country", "tier", "cand_ret", "par_ret").agg(pl.len(), pl.col("y").mean().round(4).alias("prec"),
          pl.col("parent").is_null().mean().round(4).alias("noncopy")).sort("country", "tier", "cand_ret", "par_ret"))
    pop = C.filter(~pl.col("cand_ret") & ~(pl.col("par_ret") & ~pl.col("y")))
    excl = C.filter(~pl.col("cand_ret") & pl.col("par_ret") & ~pl.col("y"))
    print(f"\nPOPULATION (cand not retrieved, parent not retrieved): {pop.height}; excluded (parent retrieved elsewhere): {excl.height}")
    print(pop.group_by("country", "tier").agg(pl.len(), pl.col("y").mean().round(4).alias("prec"),
          pl.col("parent").is_null().mean().round(4).alias("noncopy"), pl.col("allcap").mean().round(3).alias("allcap_share")).sort("country", "tier"))
    print("DF-cap subset (every key token in >3000 S1 names of the country):")
    print(pop.filter(pl.col("allcap")).group_by("country", "tier").agg(pl.len(), pl.col("y").mean().round(4).alias("prec"),
          pl.col("parent").is_null().mean().round(4).alias("noncopy")).sort("country", "tier"))
    # ---- excluded group (true parent retrieved, cand is another S1): in test only if production rejected the parent
    W_EX = float(os.environ.get("W_EX", "0.55"))
    val_production_reject_rate(excl)
    tr_df = pl.concat([pop.with_columns(pl.lit(1.0).alias("w")), excl.with_columns(pl.lit(W_EX).alias("w"))])
    # ---- grouped 5-fold OOF + cross-fitted isotonic calibration
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    y = tr_df["y"].cast(pl.Int8).to_numpy(); Xp = X(tr_df); wt = tr_df["w"].to_numpy()
    fold = (tr_df["s"].hash(11) % 5).to_numpy()
    raw = np.zeros(len(y)); models = []
    for f in range(5):
        tr, te = fold != f, fold == f
        m = lgb.train(PARAMS, lgb.Dataset(Xp[tr], y[tr], weight=wt[tr]), ROUNDS)
        raw[te] = m.predict(Xp[te]); models.append(m)
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[tr], y[tr], sample_weight=wt[tr])
        oof[te] = iso.predict(raw[te])
    iso_all = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y, sample_weight=wt)
    for nm_, pr in [("raw", raw), ("iso", oof)]:
        ll = -np.average(y * np.log(np.clip(pr, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pr, 1e-6, 1)), weights=wt)
        print(f"\nOOF {nm_}: weighted logloss {ll:.4f}  AUC {roc_auc_score(y, pr, sample_weight=wt):.4f}  base rate {np.average(y, weights=wt):.4f}  (W_EX={W_EX})")
    tr_df = tr_df.with_columns(pl.Series("p", oof), pl.Series("p_raw", raw))
    pop = tr_df
    print("calibration (OOF):")
    for pc in ["p_raw", "p"]:
        print(pop.group_by(pl.col(pc).cut([0.3, 0.5, 0.63, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]).alias("bin")).agg(
            pl.len(), pl.col("w").sum().round(0).alias("wsum"), pl.col(pc).mean().round(4).alias("p_mean"),
            ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(4).alias("prec")).sort("bin"))
    # ---- decision rule on train OOF
    pop = pop.join(ntrue, on="s", how="left").with_columns(pl.col("n_true").fill_null(0)) if ntrue is not None else pop
    A = decide(pop, "p").filter(pl.col("add"))
    eg = sum(exp_gain(p, k) for p, k in zip(A["p"].to_list(), A["k_found"].to_list()))
    wp = lambda d: round(float((d["y"] * d["w"]).sum() / max(d["w"].sum(), 1e-9)), 4)  # noqa: E731
    print("weighted precision of train adds:", wp(A), " weighted count:", round(float(A["w"].sum()), 1))
    print(f"\nTRAIN adds: {A.height}  precision {A['y'].mean():.4f}  expected gain {eg:.1f} S1-units")
    print(A.group_by("country", "tier").agg(pl.len(), pl.col("w").sum().alias("wsum"), ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(4).alias("prec"), pl.col("p").mean().round(4).alias("p_mean"),
          pl.col("parent").is_null().mean().round(4).alias("noncopy")).sort("country", "tier"))
    print("adds in DF-cap subset:", A.filter(pl.col("allcap")).height, "weighted precision", wp(A.filter(pl.col("allcap"))))
    print("adds with no candidates at all (r_npp==0):", A.filter(pl.col("r_npp") == 0).height)
    print("adds by k_found:", A.group_by(pl.col("k_found").clip(0, 4)).agg(pl.len(), ((pl.col("y") * pl.col("w")).sum() / pl.col("w").sum()).round(4).alias("prec")).sort("k_found").to_dicts())
    if ntrue is not None:
        rg = realized(A)
        print(f"realized F0.5 gain on train truth: {rg:.1f} S1-units ({rg / 1_785_000:.6f} per train S1)")
        print("realized gain by country x tier:", dict(sorted(realized(A, ["country", "tier"]).items())))
        print("realized gain by k_found:", dict(sorted(realized(A.with_columns(pl.col("k_found").clip(0, 4).alias("kb")), ["kb"]).items())))
        print("realized gain by p bin:", dict(sorted(realized(A.with_columns(pl.col("p").cut([0.8, 0.9, 0.95]).cast(pl.String).alias("pb")), ["pb"]).items())))
    KEEP = {("France", 0)}
    if ntrue is not None:   # final rule: keep (country, tier) groups with realized train gain > 3 S1-units; France tier 0 only
        rg_ct = realized(A, ["country", "tier"])
        KEEP |= {k for k, v in rg_ct.items() if v > 3.0}
        print("KEEP groups:", sorted(KEEP))
        F = A.filter(pl.struct("country", "tier").map_elements(lambda r: (r["country"], r["tier"]) in KEEP, return_dtype=pl.Boolean))
        print(f"FINAL-RULE train adds {F.height}: weighted precision {wp(F)}, realized {realized(F):.1f} S1-units, expected "
              f"{sum(exp_gain(p, k) for p, k in zip(F['p'].to_list(), F['k_found'].to_list())):.1f}")
        for c in ["India", "US"]:
            for cap in [True, False]:
                Fc = F.filter((pl.col("country") == c) & (pl.col("allcap") == cap))
                print(f"  {c} DF-cap={cap}: adds {Fc.height}, weighted precision {wp(Fc)}, non-copy share {Fc['parent'].is_null().mean():.4f}, realized {realized(Fc):.1f}")
        popc = pop.filter(pl.col("w") == 1.0)
        print("population non-copy share (parent null):", round(float(popc["parent"].is_null().mean()), 4),
              " DF-cap subset:", round(float(popc.filter(pl.col("allcap"))["parent"].is_null().mean()), 4))
    imp = m.feature_importance("gain")
    print("importance:", sorted(zip(FEATS, np.round(imp / imp.sum(), 3)), key=lambda t: -t[1])[:15])
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply(models, iso_all, KEEP)


def val_production_reject_rate(excl):
    """How often does production reject the retrieved true parent for excluded-type records? (validation, unc_val)"""
    try:
        ids = pl.read_parquet("local_data/train_source1.parquet", columns=["entity_id"]).with_row_index("parent").with_columns(pl.col("parent").cast(pl.UInt32))
        e = excl.join(ids, on="parent").with_columns((pl.col("entity_id_right").hash(42) % 10 == 0).alias("is_val"))
        u = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
        ev = e.filter(pl.col("is_val")).join(u.rename({"s1": "best"}), on="q", how="left")
        print(f"excluded records in validation: {ev.height}; in uncertain zone: {ev['prob'].is_not_null().sum()} "
              f"(best==parent {int((ev['best'] == ev['parent']).sum())}); uncertain prob mean {ev['prob'].mean()}; share prob<0.75: "
              f"{float((ev['prob'] < 0.75).sum()) / max(ev.height, 1):.3f}")
    except Exception as ex:  # informational only
        print("reject-rate check failed:", ex)


def apply(models, iso, KEEP):
    T = feats(pl.read_parquet(os.path.join(SCR, "unc_test.parquet")))
    print("\nTEST unique-best candidates:", T.height)
    print(T.group_by("country", "tier", "cand_ret", "assigned").len().sort("country", "tier", "cand_ret", "assigned"))
    T = T.filter(~pl.col("cand_ret") & ~pl.col("assigned"))
    raw = np.mean([m.predict(X(T)) for m in models], axis=0)
    T = T.with_columns(pl.Series("p_raw", raw), pl.Series("p", iso.predict(raw)), (pl.col("r_npp") == 0).alias("no_cand"))
    print("test p distribution by country:", T.group_by("country").agg(pl.len(), (pl.col("p") > 0.63).sum().alias("p>0.63"), (pl.col("p") > 0.85).sum().alias("p>0.85")).sort("country").to_dicts())
    print("test population (unassigned, cand not in its candidate list):", T.height, " of which in no candidate list:", int(T["no_cand"].sum()))
    # only (country, tier) groups with positive realized gain on train truth (France: exact tier only, no truth)
    T = T.filter(pl.struct("country", "tier").map_elements(lambda r: (r["country"], r["tier"]) in KEEP, return_dtype=pl.Boolean))
    A = decide(T, "p").filter(pl.col("add"))
    A = A.with_columns(pl.Series("eg", [exp_gain(p, k) for p, k in zip(A["p"].to_list(), A["k_found"].to_list())]))
    print(A.group_by("country", "tier", "no_cand").agg(pl.len(), pl.col("p").mean().round(4).alias("p_mean"), pl.col("eg").sum().round(1).alias("gain_S1"),
          pl.col("allcap").sum().alias("n_allcap")).sort("country", "tier", "no_cand"))
    tot = float(A["eg"].sum())
    print(f"TEST adds {A.height}; expected gain {tot:.1f} S1-units = {tot / N_S1_TEST:.6f} LB")
    A.select(pl.col("entity_id").alias("m"), pl.col("s1_id").alias("s1"), "p", "country", "tier", "no_cand", "k_found", "eg",
             "business_name", "s_name").write_parquet(os.path.join(SCR, "uncapped_adds_full.parquet"))
    A.select(pl.col("entity_id").alias("m"), pl.col("s1_id").alias("s1"), "p", "country").write_parquet(os.path.join(SCR, "uncapped_adds.parquet"))
    print(A.sample(min(30, A.height), seed=1).select("country", "business_name", "s_name", "p", "k_found", "tier"))


if __name__ == "__main__":
    main()
