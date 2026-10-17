"""Blend stage-3 probabilities with cross-encoder scores in the uncertain zone.

dump:  per-pair stage-3 probability p3 for validation pairs (train split) and all test pairs, from saved models.
blend: logistic regression on validation half A pairs that have a cross-encoder score:
           logit(p) = a * logit(p3) + b * logit(pce) + c
       pairs without a cross-encoder score keep p3. The decision rule is then chosen on half A; half B is reported.
python -m ber.blend dump  --data /work/data --work /work/run7
python -m ber.blend blend --data /work/data --work /work/run7 --ce /work/ce_out --out /work/output_v8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from ber.pipeline import NPROC, SEED, _best_per_query, hidden_s1, labels, log
from ber.stage2 import _chosen, _collective, _collective_from, _load, _macro_hard, _p1, _write_outputs, _x2, _x3


def _p3(work: Path, split: str, d: pl.DataFrame) -> pl.DataFrame:
    import lightgbm as lgb
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    s0, s1m = (lgb.Booster(model_file=str(work / f"s2_fold{k}.txt")) for k in (0, 1))
    for m in (m0, m1, s0, s1m):
        m.best_iteration = 0
    s3 = lgb.Booster(model_file=str(work / "s3_model.txt"))
    rec = pl.read_parquet(work / split / "records.parquet", columns=["name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _collective(_p1(d, m0, m1), rec)
    x = _x2(d)
    fold = d["fold"].to_numpy()
    p = np.empty(d.height, np.float32)
    for k, m in ((0, s1m), (1, s0)):
        idx = np.flatnonzero(fold == k)
        p[idx] = m.predict(x[idx], num_threads=NPROC)
    d = _collective_from(d.with_columns(pl.Series("p2", p)), "p2", "_r2", rec)
    return d.with_columns(pl.Series("p3", s3.predict(_x3(d), num_threads=NPROC).astype(np.float32)))


def eval_queries(data: Path, work: Path) -> pl.Series:
    """Queries that must be scored to evaluate validation S1 entities honestly: every record with at least one
    candidate among validation S1s (training-side orphans create false matches on validation S1s too)."""
    lab = labels(data, work)
    s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))["rid"]
    d = _load(work, "train").select("q", "s1")
    return d.filter(pl.col("s1").is_in(s1v.cast(d["s1"].dtype).implode()))["q"].unique()


def dump(data: Path, work: Path) -> None:
    lab = labels(data, work).select(pl.col("rid").alias("q"), "parent", "is_val", pl.col("src").alias("qsrc"))
    qs = eval_queries(data, work)
    d = _p3(work, "train", _load(work, "train").join(lab, on="q", how="left"))  # full context for group features
    d = d.filter(pl.col("q").is_in(qs.implode()))
    d.select("q", "s1", "p3", "is_val", (pl.col("parent") == pl.col("s1")).fill_null(False).alias("y")) \
        .write_parquet(work / "scores_val.parquet")
    rec = pl.read_parquet(work / "test" / "records.parquet", columns=["src"]).with_row_index("rid")
    t = _load(work, "test").join(rec.select(pl.col("rid").alias("q"), pl.col("src").alias("qsrc")), on="q", how="left")
    _p3(work, "test", t).select("q", "s1", "p3").write_parquet(work / "scores_test.parquet")
    log("dumped per-pair p3 for validation and test")


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def blend(data: Path, work: Path, ce: Path, out: Path, hard_k: int = 3, model: str = "lr", tag: str = "blend",
          rounds: int = 1, raw: Path | None = None, folds: int = 2, p3_runs: list | None = None,
          supp: Path | None = None, supp_min: float = 0.05) -> None:
    from sklearn.linear_model import LogisticRegression
    lab = labels(data, work).with_columns(
        (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    ha = lab.select(pl.col("rid").alias("s1"), pl.col("half_a").alias("s1_half_a"))
    ces = ce if isinstance(ce, list) else [ce]

    def attach(df: pl.DataFrame, split: str) -> pl.DataFrame:
        for i, c in enumerate(ces):
            s = pl.read_parquet(c / f"ce_scores_{split}.parquet").with_columns(
                pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).rename({"pce": f"pce{i}"})
            df = df.join(s, on=["q", "s1"], how="left")
        for i, r in enumerate(p3_runs or []):  # other stacks' stage-3 probabilities as extra stacker features
            o = pl.read_parquet(r / f"scores_{split}.parquet", columns=["q", "s1", "p3"]).with_columns(
                pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
            df = df.join(o.rename({"p3": f"raw_p3run{i}"}), on=["q", "s1"], how="left")
        if raw is not None:  # raw pair features of zone pairs (scripts/zone_raw_feats.py), for the gbm stacker
            r = pl.read_parquet(raw / f"raw_{split}.parquet")
            df = df.join(r.rename({c: f"raw_{c}" for c in r.columns if c not in ("q", "s1")}), on=["q", "s1"], how="left")
        return df.with_columns(pl.col("pce0").alias("pce"))

    def feats(df: pl.DataFrame) -> np.ndarray:
        """logit p3, per cross-encoder: logit score, margin to the best other candidate of the same record, top flag;
        plus the stage-3 margin within the record."""
        d = df.with_columns((pl.col("p3") - pl.col("p3").max().over("q")).alias("_m3"))
        cols = [_logit(d["p3"].to_numpy())]
        for i in range(len(ces)):
            c = f"pce{i}"
            d = d.with_columns(pl.col(c).fill_null(0.0).alias("_c"))
            d = d.with_columns(pl.col("_c").rank("ordinal", descending=True).over("q").alias("_r"),
                               pl.col("_c").max().over("q").alias("_top"))
            second = d.filter(pl.col("_r") == 2).select("q", pl.col("_c").alias("_sec"))
            d = d.drop([x for x in ("_sec",) if x in d.columns]).join(second, on="q", how="left")
            # margin to the best *other* candidate of the same record (positive only for the record's top pick)
            d = d.with_columns(pl.when(pl.col("_r") == 1).then(pl.col("_c") - pl.col("_sec").fill_null(0.0))
                                 .otherwise(pl.col("_c") - pl.col("_top")).alias("_marg"))
            cols += [_logit(d[c].fill_null(0.5).to_numpy()), d["_marg"].to_numpy(), (d["_r"] == 1).to_numpy().astype(float)]
        cols.append(d["_m3"].to_numpy())
        return np.column_stack(cols)

    def feats_gbm(df: pl.DataFrame) -> np.ndarray:
        """Linear-blend features plus context the logistic blend cannot combine: ranks, candidate counts, and
        cross-encoder mass per S1 / per record (collective evidence on the blended scale)."""
        d = df.with_columns(pl.len().over("q").alias("_nq"), pl.len().over("s1").alias("_ns"),
                            pl.col("p3").rank("ordinal", descending=True).over("q").alias("_r3"),
                            pl.col("p3").sum().over("s1").alias("_s3"))
        extra = [d["_nq"].to_numpy(), d["_ns"].to_numpy(), d["_r3"].to_numpy(), d["_s3"].to_numpy()]
        for i in range(len(ces)):
            c = pl.col(f"pce{i}").fill_null(0.0)
            e = d.select(c.sum().over("s1").alias("a"), (c > 0.5).sum().over("s1").alias("b"),
                         (c > 0.5).sum().over("q").alias("c"),
                         pl.col(f"pce{i}").is_null().cast(pl.Float32).alias("n"))
            extra += [e[x].to_numpy() for x in ("a", "b", "c", "n")]
        extra += [df[c].cast(pl.Float32).to_numpy() for c in df.columns if c.startswith("raw_")]
        if os.environ.get("BLEND_CE_AGREE") == "1" and len(ces) > 1:  # agreement across cross-encoders (logit scale)
            z = np.column_stack([_logit(df[f"pce{i}"].fill_null(0.5).to_numpy()) for i in range(len(ces))])
            zq = pl.DataFrame({"q": df["q"], "zm": z.mean(1)}).with_columns(
                (pl.col("zm") - pl.col("zm").max().over("q")).alias("gap"), pl.col("zm").rank("ordinal", descending=True).over("q").alias("r"))
            extra += [z.mean(1), z.std(1), z.min(1), z.max(1), zq["gap"].to_numpy(), zq["r"].to_numpy()]
        return np.column_stack([feats(df)] + extra).astype(np.float32)

    v = attach(pl.read_parquet(work / "scores_val.parquet"), "val").join(
        ha.with_columns(pl.col("s1").cast(pl.UInt32)), on="s1", how="left")
    s1va = lab.filter((pl.col("src") == 1) & pl.col("is_val") & pl.col("half_a"))["rid"].cast(pl.UInt32)
    fitmask = (v["pce"].is_not_null() & v["s1"].is_in(s1va.implode())).to_numpy()
    if os.environ.get("BLEND_DROP_H") == "1":
        # test-like stacker: leave out records whose raw parent is a hidden S1 (copies of absent businesses exist only
        # because of the 19% hiding simulation; test's extra unmatched records are decoys instead)
        rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
        hid = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19)).select(pl.col("entity_id").alias("s1"))
        H = pl.read_parquet(data / "train_pairs.parquet").join(hid, on="s1", how="semi").select(pl.col("m").alias("entity_id"))               .join(rec.select("entity_id", "rid"), on="entity_id")["rid"].cast(pl.UInt32)
        dropped = fitmask & v["q"].is_in(H.implode()).to_numpy()
        fitmask &= ~dropped
        log(f"BLEND_DROP_H: {int(dropped.sum())} half-A zone pairs of {H.len()} H records left out of the stacker fit")
    wfit = np.ones(v.height)
    if float(os.environ.get("BLEND_ORPH_W", "1")) != 1.0:
        # test has about twice the decoy density of raw training data: weight pairs of unmatched records up
        orph_q = lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].cast(pl.UInt32)
        isorph = v["q"].is_in(orph_q.implode()).to_numpy()
        wfit[isorph] = float(os.environ["BLEND_ORPH_W"])
        log(f"BLEND_ORPH_W: {int((isorph & fitmask).sum())} half-A zone pairs of unmatched records weighted {wfit[isorph][:1]}")
    y = v["y"].to_numpy()
    if model == "gbm":
        import lightgbm as lgb
        prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=NPROC)
        prm.update(json.loads(os.environ.get("BLEND_PARAMS", "{}")))
        # k-fold cross-fit over half-A S1 groups: half-A pairs get out-of-fold probabilities (so the decision rule is
        # chosen on honest scores), half B and test get the average of the fold models
        gfold = (v["s1"].hash(SEED + 11) % folds).to_numpy()
        nullce = v["pce"].is_null().to_numpy()

        def group_feats(df: pl.DataFrame, p: np.ndarray) -> np.ndarray:
            """Collective evidence recomputed on the previous round's blended probability."""
            d = df.select("q", "s1").with_columns(pl.Series("_p", p.astype(np.float32)))
            d = d.with_columns((pl.col("_p").sum().over("s1") - pl.col("_p")).alias("a"),
                               pl.col("_p").max().over("s1").alias("b"), (pl.col("_p") > 0.5).sum().over("s1").alias("c"),
                               (pl.col("_p") - pl.col("_p").max().over("q")).alias("m"),
                               pl.col("_p").rank("ordinal", descending=True).over("s1").alias("r"))
            return np.column_stack([_logit(p)] + [d[c].to_numpy() for c in "abcmr"]).astype(np.float32)

        stages, X, pv = [], feats_gbm(v), None
        for rnd in range(rounds):
            if rnd:
                X = np.column_stack([X, group_feats(v, np.where(nullce, v["p3"].to_numpy(), pv))])
            oof, models = np.full(v.height, np.nan), []
            for k in range(folds):
                tr, va = fitmask & (gfold != k), fitmask & (gfold == k)
                m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], weight=wfit[tr]), 3000,
                              valid_sets=[lgb.Dataset(X[va], y[va], weight=wfit[va])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
                models.append(m)
            pv = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
            pv[fitmask] = oof[fitmask]
            stages.append(models)
            log(f"gbm round {rnd}: iters {[m.best_iteration for m in models]}")
        coef = {"model": "gbm", "rounds": rounds, "iters": [[m.best_iteration for m in s] for s in stages],
                "n_ce": len(ces)}
        log(f"gbm blend cross-fitted on {int(fitmask.sum())} half-A zone pairs: {coef}")

        def predict(df: pl.DataFrame) -> np.ndarray:
            X, p, nc = feats_gbm(df), None, df["pce"].is_null().to_numpy()
            for rnd, models in enumerate(stages):
                if rnd:
                    X = np.column_stack([X, group_feats(df, np.where(nc, df["p3"].to_numpy(), p))])
                p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
            return p
    else:
        Xall = feats(v)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xall[fitmask], y[fitmask])
        coef = {"coef": [float(x) for x in lr.coef_[0]], "intercept": float(lr.intercept_[0]), "n_ce": len(ces)}
        log(f"blend fitted on {int(fitmask.sum())} half-A zone pairs: {coef}")

        def predict(df: pl.DataFrame) -> np.ndarray:
            return lr.predict_proba(feats(df))[:, 1]

        pv = lr.predict_proba(Xall)[:, 1]

    def apply(df: pl.DataFrame, pb: np.ndarray | None = None) -> pl.DataFrame:
        pb = predict(df) if pb is None else pb
        return df.with_columns(pl.Series("prob", np.where(df["pce"].is_null().to_numpy(), df["p3"].to_numpy(), pb)
                                         .astype(np.float32)))

    v = apply(v, pv)
    s1_all = lab.filter(pl.col("src") == 1)
    s1_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    orphans = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    grid = ([{"mode": "thr", "thr": round(float(t), 2)} for t in np.arange(0.5, 0.96, 0.05)] +
            [{"mode": "efopt", "miss": 0.0, "gamma": g} for g in json.loads(os.environ.get("BLEND_GAMMAS", "[0.8, 1.0, 1.5, 2.0, 3.0]"))])
    rep: dict = {"blend": coef}

    def with_supp(df: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
        """Add supplementary empty-address pairs (scripts/supp_empty.py); reference rows (a record's best current
        candidate) only for records the blend has not scored."""
        s = s.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("prob").cast(pl.Float32))
        new = s.filter(~pl.col("exist") & (pl.col("prob") >= supp_min)).select("q", "s1", "prob")
        ref = s.filter(pl.col("exist") & ~pl.col("q").is_in(df["q"].unique().implode())).select("q", "s1", "prob")
        return pl.concat([df.select("q", "s1", pl.col("prob").cast(pl.Float32)), new, ref])

    variants = [("p3", "p3"), ("blend", "prob")] + ([("blend_supp", "prob")] if supp is not None else [])
    for name, col in variants:
        vv = v.with_columns(pl.col(col).alias("prob"))
        if name == "blend_supp":
            vv = with_supp(vv, pl.read_parquet(supp / "supp_val.parquet"))
        best = _best_per_query(vv)
        if name == variants[-1][0]:
            best.write_parquet(work / f"val_best_{tag}.parquet")  # for error analysis
        res = {}
        for rule, k in (("plain", 1), ("hard", hard_k)):
            top = max(((_macro_hard(s1_a, truth, _chosen(best, c), orphans, k), c) for c in grid), key=lambda x: x[0])
            ch = _chosen(best, top[1])
            res[rule] = {"cfg": top[1], "valB": _macro_hard(s1_b, truth, ch, orphans, 1),
                         "hard_valB": _macro_hard(s1_b, truth, ch, orphans, hard_k)}
        rep[name] = res
        log(f"VAL {name}: plain-rule valB={res['plain']['valB']:.5f} (hard {res['plain']['hard_valB']:.5f}) | "
            f"hard-rule valB={res['hard']['valB']:.5f} (hard {res['hard']['hard_valB']:.5f})")
    use = variants[-1][0]
    cfg = rep[use]["hard"]["cfg"]
    rep["best"] = dict(cfg, use=use)
    # keep the reported "blend" entry the final one for downstream readers (boot_compare reads ["blend"]["hard"])
    rep["blend_nosupp"], rep["blend"] = rep["blend"], rep[use]
    (work / f"val_report_{tag}.json").write_text(json.dumps(rep, indent=1))
    t = apply(attach(pl.read_parquet(work / "scores_test.parquet"), "test"))
    if supp is not None:
        t = with_supp(t, pl.read_parquet(supp / "supp_test.parquet"))
    rec = pl.read_parquet(work / "test" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
    best = _best_per_query(t)
    best.write_parquet(work / f"test_best_{tag}.parquet")  # per-record best pair and probability, for analysis
    _write_outputs(work, out, cfg, best, t.select("q", "s1"), rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "blend"])
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ce", type=Path, nargs="+", default=[Path("/work/ce_out")])
    ap.add_argument("--out", type=Path, default=Path("/work/output_v8"))
    ap.add_argument("--model", choices=["lr", "gbm"], default="lr")
    ap.add_argument("--rounds", type=int, default=1, help="gbm only: stacking rounds with collective features")
    ap.add_argument("--raw", type=Path, help="gbm only: dir with raw_val/raw_test zone pair features")
    ap.add_argument("--folds", type=int, default=2, help="gbm only: cross-fit folds over half-A S1 groups")
    ap.add_argument("--p3_runs", type=Path, nargs="*", help="gbm only: other runs whose p3 become stacker features")
    ap.add_argument("--supp", type=Path, help="dir with supp_val/supp_test from scripts/supp_empty.py")
    ap.add_argument("--tag", default="blend", help="suffix of the saved validation report / best-pair files")
    a = ap.parse_args()
    if a.stage == "dump":
        dump(a.data, a.work)
    else:
        blend(a.data, a.work, a.ce, a.out, model=a.model, tag=a.tag, rounds=a.rounds, raw=a.raw, folds=a.folds, p3_runs=a.p3_runs,
              supp=a.supp)
