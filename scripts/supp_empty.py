"""Supplementary candidates for S2/S3 records with an empty address.

The main name retriever drops S1-side tokens shared by more than DF_CAP (3000) S1 records, so names made of common
words ("Surname Capital", "Surname Mountain Inc") have almost nothing left to match on and their true S1 is often never
retrieved (93% of never-retrieved true pairs have an empty address). Here, for empty-address records only, names are
searched again with a much higher cap; pairs that are not already candidates are scored by a small LightGBM trained on
non-validation records (validation half A for early stopping) and handed to the blend (ber.blend --supp).

python scripts/supp_empty.py --data /work/data --base /work/base --exist /work/supp --out /work/supp
Inputs: base/{train,test}/records.parquet, base/config.json (hide_pct), exist/exist_{train,test}.parquet (current
candidates q, s1, cos_n, score, pA). Outputs: out/supp_val.parquet (new pairs on validation S1s, plus each affected
record's best current candidate as a reference row, exist=True) and out/supp_test.parquet (q, s1, prob, exist).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import NPROC, N_FEAT, SEED, _docs, _rowdot, _tfidf, _topk, hide_pct, hidden_s1, labels, log  # noqa: E402

K, CAP = 20, 50_000
FEATS = ["cos_s", "rank_s", "top_s", "gap_s", "n_close", "r_ratio", "r_tsr", "r_pr", "r_jw", "r_skel", "n_same_s1",
         "n_same_q", "len_q", "len_s", "ex_n", "ex_pa", "ex_cos", "ex_score", "claims", "q_src3"]


def retrieve(rec: pl.DataFrame, pct: int) -> pl.DataFrame:
    out = []
    for country in rec["country"].unique().sort().to_list():
        r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))
        if pct:
            r1 = r1.filter(~hidden_s1(pl.col("entity_id"), pct))
        r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1) & pl.col("addr_empty"))
        if r1.height == 0 or r2.height == 0:
            continue
        import scipy.sparse as sp
        _, n1 = _docs(r1)
        _, n2 = _docs(r2)
        N1, N2, _ = _tfidf(n1, n2)
        df1 = np.bincount(N1.indices, minlength=N_FEAT)
        NT = (N1 @ sp.diags((df1 <= CAP).astype(np.float32))).tocsr()
        NT.eliminate_zeros()
        NT = NT.T.tocsr()
        parts, step = [], 20_000
        for i in range(0, N2.shape[0], step):
            parts.append(_topk(N2[i:i + step], NT, K, "rank_s").with_columns(pl.col("ql") + i))
            log(f"  {country}: {min(i + step, N2.shape[0])}/{N2.shape[0]}")
        u = pl.concat(parts)
        qi, si = u["ql"].to_numpy(), u["sl"].to_numpy()
        d = pl.DataFrame({"q": r2["rid"].to_numpy()[qi], "s1": r1["rid"].to_numpy()[si],
                          "cos_s": _rowdot(N2, N1, qi, si), "rank_s": u["rank_s"]})
        same = r1.group_by("name_n").len()
        d = d.join(r1.select(pl.col("rid").alias("s1"), "name_n"), on="s1").join(
            same.rename({"len": "n_same_s1"}), on="name_n").drop("name_n")
        d = d.join(r2.select(pl.col("rid").alias("q"), "name_n"), on="q").join(
            same.rename({"len": "n_same_q"}), on="name_n", how="left").drop("name_n").with_columns(
            pl.col("n_same_q").fill_null(0))
        out.append(d)
        log(f"retrieve {country}: S1={r1.height} empty-address queries={r2.height} pairs={d.height}")
    return pl.concat(out)


def features(d: pl.DataFrame, rec: pl.DataFrame, exist: pl.DataFrame) -> pl.DataFrame:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    d = d.with_columns(pl.col("cos_s").max().over("q").alias("top_s")).with_columns(
        (pl.col("top_s") - pl.col("cos_s")).alias("gap_s"),
        (pl.col("cos_s") >= pl.col("cos_s").max().over("q") - 0.02).sum().over("q").alias("n_close"))
    nm = rec.select(pl.col("rid").cast(pl.UInt32), "name_n", "name_sk", "src")
    d = d.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    d = d.join(nm.rename({"rid": "q", "name_n": "qn", "name_sk": "qk", "src": "qsrc"}), on="q") \
         .join(nm.rename({"rid": "s1", "name_n": "sn", "name_sk": "sk"}).drop("src"), on="s1")
    a, b, ak, bk = d["qn"].to_list(), d["sn"].to_list(), d["qk"].to_list(), d["sk"].to_list()
    f = {n: process.cpdist(a, b, scorer=s, workers=NPROC) for n, s in
         (("r_ratio", fuzz.ratio), ("r_tsr", fuzz.token_set_ratio), ("r_pr", fuzz.partial_ratio),
          ("r_jw", JaroWinkler.normalized_similarity))}
    f["r_skel"] = process.cpdist(ak, bk, scorer=fuzz.ratio, workers=NPROC)
    d = d.with_columns(*[pl.Series(k, v.astype(np.float32)) for k, v in f.items()],
                       pl.col("qn").str.len_chars().alias("len_q"), pl.col("sn").str.len_chars().alias("len_s"),
                       (pl.col("qsrc") == 3).cast(pl.Int8).alias("q_src3")).drop("qn", "sn", "qk", "sk", "qsrc")
    ex = exist.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    d = d.join(ex.select("q", "s1"), on=["q", "s1"], how="anti")  # only pairs that are not already candidates
    qx = ex.group_by("q").agg(pl.len().alias("ex_n"), pl.col("pA").max().alias("ex_pa"),
                              pl.col("cos_n").max().alias("ex_cos"), pl.col("score").max().alias("ex_score"))
    cl = ex.filter(pl.col("pA") > 0.5).group_by("s1").agg(pl.len().alias("claims"))
    return d.join(qx, on="q", how="left").join(cl, on="s1", how="left").with_columns(
        pl.col("ex_n").fill_null(0), pl.col("ex_pa").fill_null(0.0), pl.col("ex_cos").fill_null(0.0),
        pl.col("ex_score").fill_null(0.0), pl.col("claims").fill_null(0))


def best_existing(exist: pl.DataFrame, qs: pl.Series) -> pl.DataFrame:
    """Reference row per record: its best current candidate by stage-A probability (used only when the blend has no
    score for that record, i.e. training-side records whose new candidate is a validation S1)."""
    e = exist.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).filter(pl.col("q").is_in(qs.implode()))
    return e.sort("pA", descending=True).group_by("q", maintain_order=True).first().select(
        "q", "s1", pl.col("pA").cast(pl.Float32).alias("prob"), pl.lit(True).alias("exist"))


def main() -> None:
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--base", type=Path, default=Path("/work/base"))
    ap.add_argument("--exist", type=Path, default=Path("/work/supp"))
    ap.add_argument("--out", type=Path, default=Path("/work/supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["entity_id", "src", "country", "name_n", "addr_n", "name_sk", "addr_empty"]
    pct = hide_pct(a.base)
    lab = labels(a.data, a.base).with_columns(
        (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    frames = {}
    for split in ("train", "test"):
        rec = pl.read_parquet(a.base / split / "records.parquet", columns=cols).with_row_index("rid")
        exist = pl.read_parquet(a.exist / f"exist_{split}.parquet")
        d = features(retrieve(rec, pct if split == "train" else 0), rec, exist)
        frames[split] = (d, exist)
        log(f"{split}: {d.height} new pairs for {d['q'].n_unique()} empty-address records")
    d, exist = frames["train"]
    s1lab = lab.filter(pl.col("src") == 1).select(pl.col("rid").cast(pl.UInt32).alias("s1"), pl.col("is_val").alias("s1_val"),
                                                  pl.col("half_a").alias("s1_a"))
    qlab = lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32), pl.col("is_val").alias("q_val"))
    d = d.join(s1lab, on="s1", how="left").join(qlab, on="q", how="left").with_columns(
        (pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
    trn = d.filter(~pl.col("s1_val") & ~pl.col("q_val"))
    vaa = d.filter(pl.col("s1_val") & pl.col("s1_a"))
    log(f"train pairs {trn.height} (pos {int(trn['y'].sum())}), valA pairs {vaa.height} (pos {int(vaa['y'].sum())})")
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, num_threads=NPROC)
    X = lambda f: f.select(FEATS).cast(pl.Float32).to_numpy()  # noqa: E731
    m = lgb.train(prm, lgb.Dataset(X(trn), trn["y"].to_numpy()), 3000,
                  valid_sets=[lgb.Dataset(X(vaa), vaa["y"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"supp model: {m.best_iteration} trees, valA logloss {m.best_score['valid_0']['binary_logloss']:.4f}")
    m.save_model(str(a.out / "supp_model.txt"), num_iteration=m.best_iteration)
    v = d.filter(pl.col("s1_val"))
    v = v.select("q", "s1", pl.Series("prob", m.predict(X(v), num_iteration=m.best_iteration).astype(np.float32)),
                 pl.lit(False).alias("exist"), "y")
    ref = best_existing(exist, v["q"].unique())
    pl.concat([v.drop("y"), ref]).write_parquet(a.out / "supp_val.parquet")
    for thr in (0.1, 0.3, 0.5, 0.7):
        s = v.filter(pl.col("prob") >= thr)
        log(f"val new pairs prob>={thr}: {s.height}, true {int(s['y'].sum())}")
    t, texist = frames["test"]
    t = t.select("q", "s1", pl.Series("prob", m.predict(X(t), num_iteration=m.best_iteration).astype(np.float32)),
                 pl.lit(False).alias("exist"))
    t.write_parquet(a.out / "supp_test.parquet")
    log(f"test new pairs {t.height}, prob>=0.5: {t.filter(pl.col('prob') >= 0.5).height}")


if __name__ == "__main__":
    main()
