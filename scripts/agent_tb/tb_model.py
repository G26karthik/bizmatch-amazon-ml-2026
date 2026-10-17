"""Agent TB step 2: LightGBM tie-breaker over ambiguous empty-address one-op candidate sets.

Fit/eval halves by crc32(record id) % 2 (both directions). Records whose truth is outside the candidate set (or noise)
count as wrong picks. Reports top-1 accuracy and precision/coverage at probability thresholds; applies to TEST
unassigned records (fr27) and writes local_data/work/agent_tb_adds.parquet.

uv run --no-project --with polars==1.44.2 --with rapidfuzz --with lightgbm --with numpy python scripts/agent_tb/tb_model.py
"""
import sys
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(250)
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
META = ["m", "s1", "country", "kind", "label", "has_truth", "fr27_s1", "fold"]


def add_group_feats(d):
    g = "m"
    return d.with_columns(
        r_ldf_chg=pl.col("ldf_chg").rank("dense", descending=True).over(g),
        d_ldf_chg=pl.col("ldf_chg") - pl.col("ldf_chg").max().over(g),
        d2_ldf_chg=pl.col("ldf_chg") - pl.col("ldf_chg").min().over(g),
        r_c_all=pl.col("c_all").rank("dense", descending=True).over(g),
        sh_c_all=pl.col("c_all") / (pl.col("c_all").sum().over(g) + 1e-9),
        d_c_all=pl.col("c_all") - pl.col("c_all").max().over(g),
        d_c_same=pl.col("c_same") - pl.col("c_same").max().over(g),
        d_c_emp=pl.col("c_emp") - pl.col("c_emp").max().over(g),
        d_ratio=pl.col("ratio") - pl.col("ratio").max().over(g),
        d_tsort=pl.col("tsort") - pl.col("tsort").max().over(g),
        d_pratio=pl.col("pratio") - pl.col("pratio").max().over(g),
        r_ratio=pl.col("ratio").rank("dense", descending=True).over(g),
        d_ntok_s=pl.col("ntok_s") - pl.col("ntok_s").max().over(g),
        n_leg_eq=pl.col("leg_eq").sum().over(g),
        n_first_eq=pl.col("first_eq").sum().over(g),
        n_order_eq=pl.col("order_eq").sum().over(g),
        n_case_eq=pl.col("case_ieq_loose").sum().over(g),
        n_addr_emp=pl.col("s_addr_emp").sum().over(g),
        n_chg_in_addr=pl.col("chg_in_addr").sum().over(g),
        n_same_op=pl.len().over(g, "op_id"),
    )


def top_picks(d, pcol):
    d = d.with_columns(psum=pl.col(pcol).sum().over("m"))
    d = d.with_columns(pn=pl.col(pcol) / pl.col("psum"))
    return (d.sort(["m", pcol], descending=[False, True]).group_by("m", maintain_order=True)
            .agg(pl.all().first(), p2=pl.col(pcol).slice(1, 1).first()))


def table(t, score, thrs=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    N = t.height
    out = []
    for th in thrs:
        s = t.filter(pl.col(score) >= th)
        out.append((score, th, s.height, round(s.height / N, 4), round(s["label"].mean(), 4) if s.height else None))
    return pl.DataFrame(out, schema=["score", "thr", "picks", "coverage", "precision"], orient="row")


def main():
    tr = pl.read_parquet("local_data/work/tb_pairs_train.parquet")
    tr = tr.with_columns(fold=pl.col("m").map_elements(lambda x: zlib.crc32(x.encode()) % 2, return_dtype=pl.Int64))
    tr = add_group_feats(tr)
    feats = [c for c in tr.columns if c not in META]
    print("features", len(feats), feats)
    rec = tr.group_by("m").agg(pl.col("country").first(), pl.col("label").any().alias("in_set"),
                               pl.col("has_truth").first(), pl.len().alias("ncand"))
    print("TRAIN ambiguous records:", rec.height, "pairs:", tr.height)
    print(rec.group_by("country").agg(pl.len(), pl.col("in_set").mean().round(4).alias("truth_in_set"),
                                      pl.col("has_truth").mean().round(4).alias("is_copy"), pl.col("ncand").mean().round(2)))
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12, seed=7)
    oof = np.zeros(tr.height)
    models = []
    for f in (0, 1):
        fit = tr.filter(pl.col("fold") == f)
        idx = np.where(tr["fold"].to_numpy() != f)[0]
        ds = lgb.Dataset(fit.select(feats).to_numpy(), fit["label"].cast(pl.Int8).to_numpy(), feature_name=feats,
                         categorical_feature=["op_id", "ctry", "src"])
        b = lgb.train(params, ds, num_boost_round=700)
        models.append(b)
        oof[idx] = b.predict(tr.filter(pl.col("fold") != f).select(feats).to_numpy())
    tr = tr.with_columns(p=pl.Series(oof))
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])[:25]
    print("top gain:", [(a, int(b)) for a, b in imp])
    # baselines
    for name, col, desc in (("max ldf_chg (most common changed word)", "ldf_chg", True), ("max c_all", "c_all", True), ("max ratio", "ratio", True)):
        tb = tr.sort(["m", col], descending=[False, desc]).group_by("m", maintain_order=True).first()
        print(f"baseline top-1 [{name}]: {tb['label'].mean():.4f}")
    t = top_picks(tr, "p")
    for f in (0, 1):
        tf = t.filter(pl.col("fold") == f)
        print(f"\n=== EVAL half {f} (model fit on other half): records {tf.height}, top-1 acc {tf['label'].mean():.4f}")
        print(pl.concat([table(tf, "p"), table(tf, "pn")]))
    print("\n=== POOLED OOF by country (score p):")
    for c in ("US", "India"):
        tc = t.filter(pl.col("country") == c)
        print(c, "records", tc.height, "top-1", round(tc["label"].mean(), 4))
        print(table(tc, "p"))
    print("\nPOOLED OOF by op of top pick (p>=0.85):")
    print(t.filter(pl.col("p") >= 0.85).group_by("country", "kind").agg(pl.len(), pl.col("label").mean().round(4)).sort("country", "kind"))
    print("POOLED OOF by ncand (p>=0.85):")
    print(t.filter(pl.col("p") >= 0.85).group_by(pl.col("ncand").clip(2, 5)).agg(pl.len(), pl.col("label").mean().round(4)).sort("ncand"))
    # choose threshold: smallest thr on a fine grid s.t. pooled precision >= TARGET in each half and each country
    grid = np.round(np.arange(0.5, 0.991, 0.01), 3)
    thr = None
    for th in grid:
        ok = True
        for f in (0, 1):
            for c in ("US", "India"):
                s = t.filter((pl.col("fold") == f) & (pl.col("country") == c) & (pl.col("p") >= th))
                if s.height >= 30 and s["label"].mean() < TARGET:
                    ok = False
        s = t.filter(pl.col("p") >= th)
        if ok and s.height and s["label"].mean() >= TARGET + 0.01:
            thr = float(th)
            break
    if thr is None:
        print("NO threshold reaches target precision; writing nothing")
        return
    s = t.filter(pl.col("p") >= thr)
    print(f"\nCHOSEN thr {thr}: pooled OOF picks {s.height} ({s.height / t.height:.4f} coverage), precision {s['label'].mean():.4f}")
    # ---- TEST
    te = pl.read_parquet("local_data/work/tb_pairs_test.parquet")
    te = add_group_feats(te)
    X = te.select(feats).to_numpy()
    te = te.with_columns(p=pl.Series((models[0].predict(X) + models[1].predict(X)) / 2))
    tt = top_picks(te, "p")
    print("\nTEST ambiguous records (all, incl. fr27-assigned):", tt.height)
    print(tt.group_by("country").agg(pl.len(), pl.col("fr27_s1").is_not_null().sum().alias("fr27_assigned"),
                                     pl.col("ncand").mean().round(2), pl.col("ntok_r").mean().round(2),
                                     (pl.col("p") >= thr).mean().round(4).alias("frac_p>=thr")).sort("country"))
    print("TRAIN comparison: ncand/ntok_r/frac_p>=thr by country")
    print(t.group_by("country").agg(pl.len(), pl.col("ncand").mean().round(2), pl.col("ntok_r").mean().round(2),
                                    (pl.col("p") >= thr).mean().round(4).alias("frac_p>=thr")))
    print("op mix of top pick, train vs test:")
    print(pl.concat([t.select("country", "kind").with_columns(split=pl.lit("train")),
                     tt.select("country", "kind").with_columns(split=pl.lit("test"))])
          .group_by("split", "country", "kind").len().with_columns(frac=(pl.col("len") / pl.col("len").sum().over("split", "country")).round(3))
          .sort("split", "country", "kind"))
    # agreement check with fr27 on assigned ambiguous records (fr27 as proxy label)
    a = te.filter(pl.col("fr27_s1").is_not_null()).group_by("m").agg(pl.col("fr27_s1").first(), pl.col("s1"))
    a = a.with_columns(fr27_in_set=pl.col("s1").list.contains(pl.col("fr27_s1")))
    ta = tt.filter(pl.col("fr27_s1").is_not_null()).join(a.select("m", "fr27_in_set"), on="m")
    ta = ta.with_columns(agree=pl.col("s1") == pl.col("fr27_s1"))
    print("TEST fr27-assigned ambiguous: fr27 pick within candidate set / model top agrees with fr27:")
    print(ta.group_by("country").agg(pl.len(), pl.col("fr27_in_set").mean().round(4),
                                     pl.col("agree").mean().round(4).alias("agree_all"),
                                     pl.col("agree").filter(pl.col("p") >= thr).mean().round(4).alias("agree_p>=thr"),
                                     (pl.col("p") >= thr).sum().alias("n_p>=thr"),
                                     pl.col("agree").filter(pl.col("fr27_in_set") & (pl.col("p") >= thr)).mean().round(4).alias("agree_inset_p>=thr"))
          .sort("country"))
    # picks for unassigned records, respecting caps (5 S2 / 6 S3 per S1) given fr27 assignments
    g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t",
                    schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    cur = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", src=pl.col("matched_entity_ids").str.slice(0, 2))
           .group_by("s1", "src").len())
    cnt = {(a_, b_): n for a_, b_, n in cur.iter_rows()}
    un = tt.filter(pl.col("fr27_s1").is_null() & (pl.col("p") >= thr)).sort("p", descending=True)
    keep, capped = [], 0
    for m, s1, c, k, p in un.select("m", "s1", "country", "kind", "p").iter_rows():
        src = m[:2]
        lim = 5 if src == "S2" else 6
        if cnt.get((s1, src), 0) >= lim:
            capped += 1
            continue
        cnt[(s1, src)] = cnt.get((s1, src), 0) + 1
        keep.append((m, s1, c, k, p))
    out = pl.DataFrame(keep, schema={"m": pl.String, "s1": pl.String, "country": pl.String, "kind": pl.String, "p": pl.Float64}, orient="row")
    print(f"\nTEST unassigned picks p>={thr}: {out.height} (dropped by cap {capped})")
    print(out.group_by("country", "kind").len().sort("country", "kind"))
    print(tt.filter(pl.col("fr27_s1").is_null()).group_by("country").agg(pl.len().alias("unassigned_ambiguous")).sort("country"))
    out.filter(pl.col("country") != "France").write_parquet("local_data/work/agent_tb_adds.parquet")
    out.filter(pl.col("country") == "France").write_parquet("local_data/work/agent_tb_adds_france_UNVALIDATED.parquet")
    t.write_parquet("local_data/work/tb_oof_train_top.parquet")
    tt.write_parquet("local_data/work/tb_test_top.parquet")


if __name__ == "__main__":
    main()
