"""Agent TB follow-up: tie-breaker for LARGE one-op candidate groups (>12 candidates; top-40 by name ratio kept).

Model fit on big + small train groups (feature ncand_full), halves by crc32(record id) % 2; eval reported on big groups.
Writes TEST picks (unassigned in fr35, US/India) to local_data/work/agent_tb_namesake.parquet if band precision >= 0.87.
"""
import sys
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts/agent_tb")
from tb_model import add_group_feats, top_picks  # noqa: E402

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(250)
META = ["m", "s1", "country", "kind", "label", "has_truth", "fr27_s1", "fr35_s1", "fr35_s1", "fold", "big"]
BANDS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def extra(d):
    g = "m"
    return d.with_columns(
        d_rx_all=pl.col("rx_all") - pl.col("rx_all").max().over(g), sh_rx_all=pl.col("rx_all") / (pl.col("rx_all").sum().over(g) + 1e-9),
        d_rl_all=pl.col("rl_all") - pl.col("rl_all").max().over(g), sh_rl_all=pl.col("rl_all") / (pl.col("rl_all").sum().over(g) + 1e-9),
        d_rx_emp=pl.col("rx_emp") - pl.col("rx_emp").max().over(g), d_rl_same=pl.col("rl_same") - pl.col("rl_same").max().over(g),
        n_rec_lk_eq=pl.col("rec_lk_eq").sum().over(g), n_case_eq_x=pl.col("case_eq").sum().over(g),
        d_s_addr_len=pl.col("s_addr_len") - pl.col("s_addr_len").max().over(g))


def fold_col(d):
    return d.with_columns(fold=pl.col("m").map_elements(lambda x: zlib.crc32(x.encode()) % 2, return_dtype=pl.Int64))


def main():
    big = pl.read_parquet("local_data/work/tb_ns_train.parquet").with_columns(big=pl.lit(True))
    small = big.head(0)
    tr = pl.concat([big, small.select(big.columns)], how="vertical_relaxed")
    tr = add_group_feats(fold_col(tr)).with_columns(lncf=pl.col("ncand_full").log1p())
    tr = extra(tr)
    feats = [c for c in tr.columns if c not in META]
    rb = tr.filter(pl.col("big")).group_by("m").agg(pl.col("country").first(), pl.col("label").any().alias("in_set"),
                                                     pl.col("has_truth").first(), pl.col("ncand_full").first())
    print("TRAIN namesake records", rb.height, "pairs", tr.filter(pl.col("big")).height)
    print(rb.group_by("country").agg(pl.len(), pl.col("in_set").mean().round(4).alias("truth_in_set"),
                                     pl.col("has_truth").mean().round(4).alias("is_copy"),
                                     pl.col("ncand_full").median().alias("med_ncand_full")).sort("country"))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14, seed=7)
    oof = np.zeros(tr.height)
    fold = tr["fold"].to_numpy()
    X = tr.select(feats).to_numpy()
    y = tr["label"].cast(pl.Int8).to_numpy()
    models = []
    for f in (0, 1):
        ds = lgb.Dataset(X[fold == f], y[fold == f], feature_name=feats, categorical_feature=["op_id", "ctry", "src"])
        b = lgb.train(params, ds, num_boost_round=500)
        models.append(b)
        oof[fold != f] = b.predict(X[fold != f])
    tr = tr.with_columns(p=pl.Series(oof))
    t = top_picks(tr.filter(pl.col("big")), "p")
    print("NAMESAKE groups top-1 acc (all big records in eval halves):", round(t["label"].mean(), 4))
    t = t.with_columns(band=pl.col("p").cut(BANDS, left_closed=True))
    print("NAMESAKE groups: marginal precision per p band, per eval half and country")
    print(t.group_by("fold", "country", "band").agg(pl.len(), pl.col("p").mean().round(3).alias("mean_p"),
                                                    pl.col("label").mean().round(4).alias("prec")).sort("fold", "country", "band"))
    print("NAMESAKE groups pooled halves: band precision / cumulative precision+coverage")
    for c in ("US", "India"):
        tc = t.filter(pl.col("country") == c)
        for lo, hi in ((0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)):
            s = tc.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
            cu = tc.filter(pl.col("p") >= lo)
            print(f"  {c} band [{lo},{hi}): n {s.height} prec {s['label'].mean() if s.height else float('nan'):.4f} | p>={lo}: n {cu.height} cov {cu.height / tc.height:.4f} prec {cu['label'].mean() if cu.height else float('nan'):.4f}")
    # threshold: lowest band edge such that every band at/above it has pooled precision >= 0.87 in each country (bands with n>=30)
    thr = None
    for lo in (0.8, 0.85, 0.9, 0.95):
        ok = True
        for c in ("US", "India"):
            tc = t.filter(pl.col("country") == c)
            for a, b_ in ((0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)):
                if a < lo:
                    continue
                for f in (0, 1):
                    s = tc.filter((pl.col("p") >= a) & (pl.col("p") < b_) & (pl.col("fold") == f))
                    if s.height >= 30 and s["label"].mean() < 0.87:
                        ok = False
        if ok:
            thr = lo
            break
    if thr is None:
        print("NOT REACHABLE: no threshold with band precision >= 0.87; writing nothing")
        return
    print("CHOSEN threshold", thr)
    te = pl.read_parquet("local_data/work/tb_ns_test.parquet")
    te = add_group_feats(te).with_columns(lncf=pl.col("ncand_full").log1p())
    te = extra(te)
    Xt = te.select(feats).to_numpy()
    te = te.with_columns(p=pl.Series((models[0].predict(Xt) + models[1].predict(Xt)) / 2))
    tt = top_picks(te, "p")
    tt.write_parquet("local_data/work/tb_ns_test_top.parquet")
    ta = tt.filter(pl.col("fr35_s1").is_not_null()).with_columns(agree=pl.col("s1") == pl.col("fr35_s1"))
    print("TEST namesake records:", tt.height, "fr35-assigned:", ta.height)
    print("agreement of top pick with fr35 on assigned big records, p>=thr:")
    print(ta.filter(pl.col("p") >= thr).group_by("country").agg(pl.len(), pl.col("agree").mean().round(4)).sort("country"))
    g = pl.read_csv("output/v28q_fr35/matching_results.tsv", separator="\t",
                    schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    asg = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", m="matched_entity_ids"))
    assigned = set(asg["m"].to_list())
    cnt = {}
    for s1, m in asg.iter_rows():
        cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
    un = tt.filter(pl.col("fr35_s1").is_null() & (pl.col("p") >= thr) & (pl.col("country") != "France")).sort("p", descending=True)
    keep, capped = [], 0
    for m, s1, c, k, p in un.select("m", "s1", "country", "kind", "p").iter_rows():
        assert m not in assigned
        lim = 5 if m[:2] == "S2" else 6
        if cnt.get((s1, m[:2]), 0) >= lim:
            capped += 1
            continue
        cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
        keep.append((m, s1, c, k, p))
    out = pl.DataFrame(keep, schema={"m": pl.String, "s1": pl.String, "country": pl.String, "kind": pl.String, "p": pl.Float64}, orient="row")
    print(f"TEST unassigned namesake picks p>={thr}: {out.height} (cap-dropped {capped})")
    print(out.group_by("country").agg(pl.len(), pl.col("p").mean().round(4)).sort("country"))
    print(out.group_by("country", "kind").len().sort("country", "kind"))
    for x in (0.85, 0.9, 0.95):
        print("  p>=", x, out.filter(pl.col("p") >= x).group_by("country").len().sort("country").rows())
    fr = tt.filter(pl.col("fr35_s1").is_null() & (pl.col("p") >= thr) & (pl.col("country") == "France"))
    print("France (not written):", fr.height)
    out.write_parquet("local_data/work/agent_tb_namesake.parquet")


if __name__ == "__main__":
    main()
