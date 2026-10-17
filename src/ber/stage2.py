"""Stage 2 on top of the pruning cascade: full training data, out-of-fold stage-1 probabilities, collective features.

1. prune:  stage-A model (work/modelA.txt) drops weak candidates for every split -> {split}/pruned/*.parquet
2. stage 1: two main models on query folds (fold = hash(q) % 2), all non-val training queries, early stopping on val-A.
            Each pair's p1 comes from the model that did not see its query (test/val use the same fold rule).
3. stage 2: collective features (how the query's mass is spread; how strongly *other* queries claim the same S1,
            overall / same source / other source) + all pair features -> final LightGBM, early stopping on val-A.
4. decide:  threshold vs expected-F0.5 chosen on val-A; honest score on val-B; train-sample F0.5 for the gap.

python -m ber.stage2 all --data /work/data --work /work/run2 --out /work/output_s2
"""
from __future__ import annotations

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import polars as pl

from ber.pipeline import (FEATS, NPROC, feat_names, PARAMS_B, SEED, _best_per_query, _parts, _score_at, _eval_block, _to_x,
                          labels, log, macro_on)

PRUNE_P2 = 0.001
S2FEATS = ["p1", "p1_rank_q", "p1_max_q", "p1_margin", "p1_share_q", "p1_am",
           "s1_sum_other", "s1_cnt_other", "s1_max_other", "s1_sum_same", "s1_sum_diff", "s1_sum_all_other",
           "s1_n_pruned", "sib_name_ratio", "sib_name_tsr", "sib_skel_ratio", "sib_addr_tsr"]


def _fold():
    return (pl.col("q").hash(SEED + 2) % 2).cast(pl.Int8).alias("fold")


def prune(work: Path, split: str) -> None:
    import lightgbm as lgb
    model_a = lgb.Booster(model_file=str(work / "modelA.txt"))
    out = work / split / "pruned"
    out.mkdir(exist_ok=True)
    kept = total = 0
    for pp in _parts(work, split):
        f = pl.read_parquet(pp)
        pa = model_a.predict(_to_x(f), num_threads=NPROC).astype(np.float32)
        g = f.with_columns(pl.Series("pA", pa)).filter(pl.col("pA") >= PRUNE_P2).with_columns(_fold())
        g = g.with_columns(pl.lit(pp.stem.rsplit("_part", 1)[0]).alias("country"))
        g.write_parquet(out / pp.name)
        kept, total = kept + g.height, total + f.height
    log(f"prune {split}: kept {kept}/{total} ({kept / total:.1%})")


def cand_min_pa(work: Path) -> float:
    """Extra candidate-budget cutoff on stage-A probability (work/config.json {"cand_min_pA": x}). The organizers rank
    smaller candidate sets higher, so the final candidate list is pA >= max(PRUNE_P2, cand_min_pA)."""
    cfg = work / "config.json"
    return float(json.loads(cfg.read_text()).get("cand_min_pA", 0.0)) if cfg.exists() else 0.0


NAMEFREQ_FEATS = ["name_exact", "skel_exact", "q_name_s1_count", "q_skel_s1_count", "s1_name_s1_count",
                  "s1_skel_s1_count"]


def _name_freq(work: Path, split: str, d: pl.DataFrame) -> pl.DataFrame:
    """How ambiguous is a name: number of S1 records in the pool (per country) with the same normalised name /
    consonant skeleton. A unique exact name is strong evidence even without an address; a common one is not."""
    from ber.pipeline import hide_pct, hidden_s1
    rec = pl.read_parquet(work / split / "records.parquet",
                          columns=["entity_id", "src", "country", "name_n", "name_sk"]).with_row_index("rid")
    pct = hide_pct(work) if split == "train" else 0
    pool = rec.filter(pl.col("src") == 1)
    if pct:
        pool = pool.filter(~hidden_s1(pl.col("entity_id"), pct))
    cn = pool.group_by("country", "name_n").len().rename({"len": "cnt_n"})
    ck = pool.group_by("country", "name_sk").len().rename({"len": "cnt_k"})
    r = (rec.select("rid", "country", "name_n", "name_sk")
            .join(cn, on=["country", "name_n"], how="left").join(ck, on=["country", "name_sk"], how="left")
            .with_columns(pl.col("cnt_n").fill_null(0), pl.col("cnt_k").fill_null(0), pl.col("rid").cast(d["q"].dtype)))
    qa = r.select(pl.col("rid").alias("q"), pl.col("name_n").alias("_qn"), pl.col("name_sk").alias("_qk"),
                  pl.col("cnt_n").alias("q_name_s1_count"), pl.col("cnt_k").alias("q_skel_s1_count"))
    sa = r.select(pl.col("rid").alias("s1"), pl.col("name_n").alias("_sn"), pl.col("name_sk").alias("_sk"),
                  pl.col("cnt_n").alias("s1_name_s1_count"), pl.col("cnt_k").alias("s1_skel_s1_count"))
    d = d.join(qa, on="q", how="left").join(sa.with_columns(pl.col("s1").cast(d["s1"].dtype)), on="s1", how="left")
    return d.with_columns((pl.col("_qn") == pl.col("_sn")).alias("name_exact"),
                          ((pl.col("_qk") == pl.col("_sk")) & (pl.col("_qk") != "")).alias("skel_exact")
                          ).drop("_qn", "_qk", "_sn", "_sk")


def h_records(work: Path, pairs: Path) -> pl.Series:
    """Training records whose raw ground-truth parent is a hidden S1: copies of absent businesses, which exist only
    because of the hiding simulation (test's extra unmatched records are decoys instead)."""
    from ber.pipeline import hide_pct, hidden_s1
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
    hid = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), hide_pct(work))).select(pl.col("entity_id").alias("s1"))
    return pl.read_parquet(pairs).join(hid, on="s1", how="semi").select(pl.col("m").alias("entity_id"))              .join(rec.select("entity_id", "rid"), on="entity_id")["rid"]


def _load(work: Path, split: str) -> pl.DataFrame:
    d = pl.concat([pl.read_parquet(p) for p in sorted((work / split / "pruned").glob("*.parquet"))])
    if cand_min_pa(work) > 0:
        d = d.filter(pl.col("pA") >= cand_min_pa(work))
    cfgp = work / "config.json"
    if split == "train" and cfgp.exists() and json.loads(cfgp.read_text()).get("drop_h"):
        # test-like universe: the hidden S1s stay out of the candidate pool, and their copies leave the data entirely
        # (training rows, group features, validation)
        H = h_records(work, Path(json.loads(cfgp.read_text())["drop_h"])).cast(d["q"].dtype)
        n0 = d.height
        d = d.filter(~pl.col("q").is_in(H.implode()))
        log(f"train: drop_h removed {n0 - d.height} pruned pairs of {H.len()} records whose parent is hidden")
    if cfgp.exists() and json.loads(cfgp.read_text()).get("name_freq"):
        d = _name_freq(work, split, d)
    ec = work / split / "emb_cos_pruned.parquet"
    if ec.exists():  # neural similarity for every pruned pair (GPU, ber.neural score)
        e = pl.read_parquet(ec).with_columns(pl.col("q").cast(d["q"].dtype), pl.col("s1").cast(d["s1"].dtype))
        d = d.drop([c for c in ("emb_cos", "emb_top1", "d_emb_top1", "emb_rank") if c in d.columns])
        d = d.join(e, on=["q", "s1"], how="left").with_columns(pl.col("emb_cos").fill_null(-1.0))
        d = d.with_columns(pl.col("emb_cos").max().over("q").alias("emb_top1"),
                           (pl.col("emb_cos").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16).alias("emb_rank"))
        d = d.with_columns((pl.col("emb_top1") - pl.col("emb_cos")).alias("d_emb_top1"))
        log(f"{split}: joined emb_cos for {e.height} pruned pairs")
    return d


def export_pruned(work: Path, split: str) -> None:
    """(q, s1) of pruned pairs, for GPU scoring on another machine."""
    pl.concat([pl.read_parquet(p, columns=["q", "s1"]) for p in sorted((work / split / "pruned").glob("*.parquet"))])         .write_parquet(work / split / "pruned_pairs.parquet")


def _collective(d: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """d has q, s1, p1, qsrc, country. Adds S2FEATS (per country windows are implicit: q/s1 never cross countries)."""
    d = d.with_columns(
        (pl.col("p1").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16).alias("p1_rank_q"),
        pl.col("p1").max().over("q").alias("p1_max_q"),
        pl.col("p1").sum().over("q").alias("_psum"),
    )
    second = d.filter(pl.col("p1_rank_q") == 1).select("q", pl.col("p1").alias("_p2"))
    d = d.join(second, on="q", how="left").with_columns(
        pl.when(pl.col("p1_rank_q") == 0).then(pl.col("p1") - pl.col("_p2").fill_null(0))
          .otherwise(pl.col("p1") - pl.col("p1_max_q")).alias("p1_margin"),
        (pl.col("p1") / pl.col("_psum").clip(lower_bound=1e-6)).alias("p1_share_q"),
        (pl.col("p1_rank_q") == 0).alias("p1_am"),
    ).drop("_p2", "_psum")
    d = d.with_columns(pl.when(pl.col("p1_am")).then(pl.col("p1")).otherwise(0.0).alias("_amp"),
                       (pl.col("p1_am") & (pl.col("p1") > 0.5)).cast(pl.Int32).alias("_c05"))
    d = d.with_columns(
        (pl.col("_amp").sum().over("s1") - pl.col("_amp")).alias("s1_sum_other"),
        (pl.col("_c05").sum().over("s1") - pl.col("_c05")).alias("s1_cnt_other"),
        (pl.col("_amp").sum().over("s1", "qsrc") - pl.col("_amp")).alias("s1_sum_same"),
        (pl.col("p1").sum().over("s1") - pl.col("p1")).alias("s1_sum_all_other"),
        pl.len().over("s1").alias("s1_n_pruned"),
        pl.col("_amp").rank("ordinal", descending=True).over("s1").alias("_amr"),
    )
    d = d.with_columns((pl.col("s1_sum_other") - pl.col("s1_sum_same")).alias("s1_sum_diff"))
    t1 = d.filter(pl.col("_amr") == 1).select("s1", pl.col("_amp").alias("_t1"), pl.col("q").alias("_q1"))
    t2 = d.filter(pl.col("_amr") == 2).select("s1", pl.col("_amp").alias("_t2"), pl.col("q").alias("_q2"))
    d = d.join(t1, on="s1", how="left").join(t2, on="s1", how="left").with_columns(
        pl.when(pl.col("_amr") == 1).then(pl.col("_t2").fill_null(0)).otherwise(pl.col("_t1").fill_null(0))
          .alias("s1_max_other"),
        pl.when(pl.col("_amr") == 1).then(pl.col("_q2")).otherwise(pl.col("_q1")).alias("_sib"),
    ).drop("_amp", "_c05", "_amr", "_t1", "_t2", "_q1", "_q2")
    return _siblings(d, rec)


def _siblings(d: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Compare each query with the strongest *other* claimer of the same S1 (e.g. an S3 record of the same business
    that has an address). Helps empty-address / garbled records join the right group. -1 when no real sibling."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    r = rec.select("rid", "name_n", "addr_n", "name_sk")
    a = d.select(pl.col("q").alias("rid")).join(r, on="rid", how="left")
    b = d.select(pl.col("_sib").alias("rid")).join(r, on="rid", how="left")
    valid = (d["_sib"].is_not_null() & (d["s1_max_other"] > 0.05)).to_numpy()
    n1, n2 = a["name_n"].fill_null("").to_list(), b["name_n"].fill_null("").to_list()
    k1, k2 = a["name_sk"].fill_null("").to_list(), b["name_sk"].fill_null("").to_list()
    d1, d2 = a["addr_n"].fill_null("").to_list(), b["addr_n"].fill_null("").to_list()
    both_addr = ((a["addr_n"].fill_null("") != "") & (b["addr_n"].fill_null("") != "")).to_numpy()
    f = {"sib_name_ratio": cpdist(n1, n2, scorer=fuzz.ratio, workers=-1),
         "sib_name_tsr": cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1),
         "sib_skel_ratio": cpdist(k1, k2, scorer=fuzz.ratio, workers=-1),
         "sib_addr_tsr": np.where(both_addr, cpdist(d1, d2, scorer=fuzz.token_set_ratio, workers=-1), -1)}
    return d.with_columns(*[pl.Series(k, np.where(valid, v, -1).astype(np.float32)) for k, v in f.items()]).drop("_sib")


def _names2(f: pl.DataFrame) -> list[str]:
    return feat_names(f) + [c for c in NAMEFREQ_FEATS if c in f.columns] + S2FEATS


def _x2(f: pl.DataFrame) -> np.ndarray:
    return f.select([pl.col(c).cast(pl.Float32) for c in _names2(f)]).to_numpy()


def _fit(x, y, xv, yv, names, tag, w=None, wv=None):
    import lightgbm as lgb
    dtr = lgb.Dataset(x, y, weight=w, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(xv, yv, weight=wv, reference=dtr, free_raw_data=True)
    ev: dict = {}
    params = dict(PARAMS_B, **json.loads(os.environ.get("LGB_OVERRIDES", "{}")))  # e.g. ensemble diversity runs
    m = lgb.train(params, dtr, num_boost_round=int(os.environ.get("MAX_ROUNDS", "3000")), valid_sets=[dtr, dva], valid_names=["train", "valA"],
                  callbacks=[lgb.early_stopping(int(os.environ.get("ES_PATIENCE", "50")), verbose=False),
                             lgb.record_evaluation(ev)])
    bi = m.best_iteration
    log(f"{tag}: rows={len(y)} best_iteration={bi} train_logloss={ev['train']['binary_logloss'][bi - 1]:.5f} "
        f"valA_logloss={ev['valA']['binary_logloss'][bi - 1]:.5f}")
    return m


def _p1(d: pl.DataFrame, m0, m1) -> pl.DataFrame:
    # Each model predicts on exactly the columns it was trained on (robust to extra feature columns in d).
    x = d.select([pl.col(c).cast(pl.Float32) for c in m0.feature_name()]).to_numpy()
    fold = d["fold"].to_numpy()
    p = np.empty(d.height, np.float32)
    for k, m in ((0, m1), (1, m0)):  # rows in fold k are scored by the model trained on the other fold
        idx = np.flatnonzero(fold == k)
        p[idx] = m.predict(x[idx], num_threads=NPROC, num_iteration=m.best_iteration)
    return d.with_columns(pl.Series("p1", p))


def train(data: Path, work: Path) -> None:
    lab = labels(data, work)
    lab = lab.with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val", "half_a", pl.col("src").alias("qsrc"))
    d = _load(work, "train").join(qlab, on="q", how="left")
    d = d.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
    log(f"pruned train pairs {d.height}, positives {d['y'].sum()}")

    trn = d.filter(~pl.col("is_val"))
    vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
    xv, yv = _to_x(vaa), vaa["y"].to_numpy().astype(np.int8)
    models = []
    for k in (0, 1):
        t = trn.filter(pl.col("fold") == k)
        models.append(_fit(_to_x(t), t["y"].to_numpy().astype(np.int8), xv, yv, feat_names(t), f"stage1 fold{k}"))
    del trn, vaa, t
    for k, m in enumerate(models):
        m.save_model(str(work / f"s1_fold{k}.txt"), num_iteration=m.best_iteration)

    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _collective(_p1(d, *models), rec)
    del rec
    trn = d.filter(~pl.col("is_val"))
    vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
    model = _fit(_x2(trn), trn["y"].to_numpy().astype(np.int8), _x2(vaa), vaa["y"].to_numpy().astype(np.int8),
                 _names2(trn), "stage2")
    model.save_model(str(work / "s2_model.txt"), num_iteration=model.best_iteration)
    del trn, vaa
    d = d.with_columns(pl.Series("p2", model.predict(_x2(d), num_threads=NPROC).astype(np.float32)))
    _evaluate(work, lab, d, model)


def evaluate(data: Path, work: Path) -> None:
    """Re-run validation from saved models (no retraining)."""
    import lightgbm as lgb
    lab = labels(data, work)
    lab = lab.with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val", "half_a", pl.col("src").alias("qsrc"))
    d = _load(work, "train").join(qlab, on="q", how="left")
    d = d.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    m0.best_iteration = m1.best_iteration = 0
    model = lgb.Booster(model_file=str(work / "s2_model.txt"))
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _collective(_p1(d, m0, m1), rec)
    del rec
    d = d.with_columns(pl.Series("p2", model.predict(_x2(d), num_threads=NPROC).astype(np.float32)))
    log("evaluate: predictions recomputed from saved models")
    _evaluate(work, lab, d, model)


def _evaluate(work: Path, lab: pl.DataFrame, d: pl.DataFrame, model) -> None:
    s1_all = lab.filter(pl.col("src") == 1)
    s1_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    s1_t = s1_all.filter(~pl.col("is_val") & (pl.col("rid").hash(SEED + 9) % 40 == 0))["rid"].to_list()
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    rep: dict = {}
    for name, col in (("stage1_oof", "p1"), ("stage2", "p2")):
        best = _best_per_query(d.with_columns(pl.col(col).alias("prob")))
        top_a, ef_a = _eval_block(f"{name}_valA", s1_a, truth, best, rep)
        cfg = ({"mode": "efopt", "miss": 0.0, "thr": top_a[1]} if ef_a > top_a[0] else {"mode": "thr", "thr": top_a[1]})
        vb, tr_s = _score_at(s1_b, truth, best, cfg), _score_at(s1_t, truth, best, cfg)
        rep[name] = {"cfg": cfg, "valB": vb, "train_sample": tr_s}
        log(f"VAL {name}: cfg={cfg} valB={vb:.5f} train_sample={tr_s:.5f}")
        if name == "stage2":
            best.filter(pl.col("s1").is_in(pl.Series(s1_a + s1_b).implode())).write_parquet(work / "val_best_s2.parquet")
    use = "stage2" if rep["stage2"]["valB"] >= rep["stage1_oof"]["valB"] else "stage1_oof"
    rep["best"] = dict(rep[use]["cfg"], use=use, f05=rep[use]["valB"])
    imp = dict(zip(model.feature_name(), model.feature_importance("gain").round().tolist()))
    rep["importance_stage2"] = dict(sorted(imp.items(), key=lambda x: -x[1])[:25])
    (work / "val_report_s2.json").write_text(json.dumps(rep, indent=1))
    log(f"CHOSEN {rep['best']}")


def _chosen(best: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Apply a decision config: fixed threshold, or expected-F0.5 subset selection on prob**gamma (gamma > 1 makes the
    selection stricter, used when test has more decoy records than training)."""
    from ber.pipeline import decide_pairs
    if cfg["mode"] == "efopt":
        g = float(cfg.get("gamma", 1.0))
        return decide_pairs(best.with_columns(pl.col("prob") ** g), cfg.get("miss", 0.0))
    return best.filter(pl.col("prob") >= cfg["thr"]).select("q", "s1")


def _macro_hard(s1_list: list, truth: dict, chosen: pl.DataFrame, orphans: set, k: int) -> float:
    """Macro F0.5 where every false match coming from an orphan (unmatched) record counts k times: validation with the
    decoy density observed on test (~3x more decoy-like orphans than the training simulation)."""
    from ber.metric import entity_f05
    pm: dict = {}
    for q, s in chosen.iter_rows():
        pm.setdefault(s, set()).add(q)
        if q in orphans:
            for j in range(1, k):
                pm[s].add(("decoy", q, j))
    return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1_list]))


def train_v5(data: Path, work: Path, src_work: Path, orphan_w: float, hard_k: int) -> None:
    """Stage 2 retrained with orphan (unmatched) records up-weighted, decision rule chosen on hard validation.
    Reuses stage-1 fold models from src_work (same candidate set)."""
    import shutil
    import lightgbm as lgb
    lab = labels(data, work)
    lab = lab.with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val", "half_a", pl.col("src").alias("qsrc"))
    d = _load(work, "train").join(qlab, on="q", how="left")
    d = d.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
    for k in (0, 1):
        if (src_work / f"s1_fold{k}.txt").resolve() != (work / f"s1_fold{k}.txt").resolve():
            shutil.copy(src_work / f"s1_fold{k}.txt", work / f"s1_fold{k}.txt")
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    m0.best_iteration = m1.best_iteration = 0
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _collective(_p1(d, m0, m1), rec)
    del rec
    d = d.with_columns(pl.when(pl.col("parent").is_null()).then(orphan_w).otherwise(1.0).alias("w"))
    trn = d.filter(~pl.col("is_val"))
    vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
    model = _fit(_x2(trn), trn["y"].to_numpy().astype(np.int8), _x2(vaa), vaa["y"].to_numpy().astype(np.int8),
                 _names2(trn), f"stage2 (orphan weight {orphan_w})", w=trn["w"].to_numpy(), wv=vaa["w"].to_numpy())
    model.save_model(str(work / "s2_model.txt"), num_iteration=model.best_iteration)
    del trn, vaa
    d = d.with_columns(pl.Series("p2", model.predict(_x2(d), num_threads=NPROC).astype(np.float32)))

    s1_all = lab.filter(pl.col("src") == 1)
    s1_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    orphans = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    best = _best_per_query(d.with_columns(pl.col("p2").alias("prob")))
    best = best.filter(pl.col("s1").is_in(pl.Series(s1_a + s1_b).implode()))
    best.write_parquet(work / "val_best_s2.parquet")
    cands = ([{"mode": "thr", "thr": round(float(t), 2)} for t in np.arange(0.5, 0.99, 0.05)] +
             [{"mode": "efopt", "miss": 0.0, "gamma": g} for g in (1.0, 1.5, 2.0, 3.0, 4.0)])
    rep: dict = {"orphan_weight": orphan_w, "hard_k": hard_k, "grid": []}
    top = None
    for cfg in cands:
        ch = _chosen(best, cfg)
        ha = _macro_hard(s1_a, truth, ch, orphans, hard_k)
        rep["grid"].append({**cfg, "hard_valA": ha})
        if top is None or ha > top[0]:
            top = (ha, cfg)
        log(f"{cfg} hard_valA={ha:.5f}")
    cfg = top[1]
    ch = _chosen(best, cfg)
    rep["best"] = dict(cfg, use="stage2", f05=_macro_hard(s1_b, truth, ch, orphans, 1),
                       hard_valB=_macro_hard(s1_b, truth, ch, orphans, hard_k))
    # reference: the same model with the plain-validation decision rule of v4
    ref = {"mode": "efopt", "miss": 0.0, "gamma": 1.0}
    rep["v4_rule"] = {"valB": _macro_hard(s1_b, truth, _chosen(best, ref), orphans, 1),
                      "hard_valB": _macro_hard(s1_b, truth, _chosen(best, ref), orphans, hard_k)}
    (work / "val_report_s2.json").write_text(json.dumps(rep, indent=1))
    log(f"CHOSEN {rep['best']} | same model, v4 rule: {rep['v4_rule']}")


# ---------------------------------------------------------------- stage 3 (iterated group reasoning)
def _collective_from(d: pl.DataFrame, col: str, suffix: str, rec: pl.DataFrame) -> pl.DataFrame:
    """Recompute the stage-2 group features from probability column `col`; new columns get `suffix`."""
    base = d.select("q", "s1", "qsrc", pl.col(col).alias("p1"))
    g = _collective(base, rec).drop("qsrc")
    g = g.rename({c: f"{c}{suffix}" for c in S2FEATS})
    return d.join(g, on=["q", "s1"], how="left")


def _s3_names(d: pl.DataFrame) -> list[str]:
    return _names2(d) + ["p2"] + [f"{c}_r2" for c in S2FEATS]


def _x3(d: pl.DataFrame) -> np.ndarray:
    return d.select([pl.col(c).cast(pl.Float32) for c in _s3_names(d)]).to_numpy()


def _oof_stage2(d: pl.DataFrame, orphan_w: float, tag: str):
    """Two stage-2 models on query folds; returns (d with out-of-fold p2, [model_fold0, model_fold1])."""
    vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
    xv, yv, wv = _x2(vaa), vaa["y"].to_numpy().astype(np.int8), vaa["w"].to_numpy()
    models = []
    for k in (0, 1):
        t = d.filter(~pl.col("is_val") & (pl.col("fold") == k))
        models.append(_fit(_x2(t), t["y"].to_numpy().astype(np.int8), xv, yv, _names2(t), f"{tag} fold{k}",
                           w=t["w"].to_numpy(), wv=wv))
    x = _x2(d)
    fold = d["fold"].to_numpy()
    p = np.empty(d.height, np.float32)
    for k, m in ((0, models[1]), (1, models[0])):
        idx = np.flatnonzero(fold == k)
        p[idx] = m.predict(x[idx], num_threads=NPROC, num_iteration=m.best_iteration)
    return d.with_columns(pl.Series("p2", p)), models


def train_s3(data: Path, work: Path, src_work: Path | None, orphan_w: float, hard_k: int) -> None:
    """Stage 1 (reused from src_work, or trained here with name-frequency features when src_work is None)
    -> out-of-fold stage 2 -> group features from p2 -> stage 3."""
    import shutil
    import lightgbm as lgb
    lab = labels(data, work)
    lab = lab.with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val", "half_a", pl.col("src").alias("qsrc"))
    d = _load(work, "train").join(qlab, on="q", how="left")
    d = d.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"),
                       pl.when(pl.col("parent").is_null()).then(orphan_w).otherwise(1.0).alias("w"))
    if src_work is None:
        names1 = feat_names(d) + ([] if os.environ.get("S1_NO_NAMEFREQ") == "1" else
                                  [c for c in NAMEFREQ_FEATS if c in d.columns])
        x1 = lambda f: f.select([pl.col(c).cast(pl.Float32) for c in names1]).to_numpy()  # noqa: E731
        vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
        xv, yv = x1(vaa), vaa["y"].to_numpy().astype(np.int8)
        for k in (0, 1):
            t = d.filter(~pl.col("is_val") & (pl.col("fold") == k))
            m = _fit(x1(t), t["y"].to_numpy().astype(np.int8), xv, yv, names1, f"stage1 fold{k} (+name freq)")
            m.save_model(str(work / f"s1_fold{k}.txt"), num_iteration=m.best_iteration)
        del vaa, t
    else:
        for k in (0, 1):
            if (src_work / f"s1_fold{k}.txt").resolve() != (work / f"s1_fold{k}.txt").resolve():
                shutil.copy(src_work / f"s1_fold{k}.txt", work / f"s1_fold{k}.txt")
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    m0.best_iteration = m1.best_iteration = 0
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _collective(_p1(d, m0, m1), rec)
    log("stage 1 probabilities + group features ready")
    d, s2m = _oof_stage2(d, orphan_w, "stage2-oof")
    for k, m in enumerate(s2m):
        m.save_model(str(work / f"s2_fold{k}.txt"), num_iteration=m.best_iteration)
    d = _collective_from(d, "p2", "_r2", rec)
    del rec
    log("stage 2 out-of-fold probabilities + second-round group features ready")
    trn = d.filter(~pl.col("is_val"))
    vaa = d.filter(pl.col("is_val") & pl.col("half_a"))
    model = _fit(_x3(trn), trn["y"].to_numpy().astype(np.int8), _x3(vaa), vaa["y"].to_numpy().astype(np.int8),
                 _s3_names(trn), "stage3", w=trn["w"].to_numpy(), wv=vaa["w"].to_numpy())
    model.save_model(str(work / "s3_model.txt"), num_iteration=model.best_iteration)
    del trn, vaa
    d = d.with_columns(pl.Series("p3", model.predict(_x3(d), num_threads=NPROC).astype(np.float32)))
    _decide_report(work, lab, d, ["p1", "p2", "p3"], hard_k, model)


def _decide_report(work: Path, lab: pl.DataFrame, d: pl.DataFrame, cols: list, hard_k: int, model) -> None:
    s1_all = lab.filter(pl.col("src") == 1)
    s1_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    s1_t = s1_all.filter(~pl.col("is_val") & (pl.col("rid").hash(SEED + 9) % 40 == 0))["rid"].to_list()
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    orphans = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    grid = ([{"mode": "thr", "thr": round(float(t), 2)} for t in np.arange(0.5, 0.96, 0.05)] +
            [{"mode": "efopt", "miss": 0.0, "gamma": g} for g in (0.8, 1.0, 1.5, 2.0, 3.0)])
    rep: dict = {"hard_k": hard_k}
    for col in cols:
        best = _best_per_query(d.with_columns(pl.col(col).alias("prob")))
        if col == cols[-1]:
            best.filter(pl.col("s1").is_in(pl.Series(s1_a + s1_b).implode())).write_parquet(work / "val_best_s2.parquet")
        res = {}
        for name, k in (("plain", 1), ("hard", hard_k)):
            top = max(((_macro_hard(s1_a, truth, _chosen(best, c), orphans, k), c) for c in grid), key=lambda x: x[0])
            ch = _chosen(best, top[1])
            res[name] = {"cfg": top[1], "valA": top[0], "valB": _macro_hard(s1_b, truth, ch, orphans, 1),
                         "hard_valB": _macro_hard(s1_b, truth, ch, orphans, hard_k),
                         "train_sample": _macro_hard(s1_t, truth, _chosen(best, top[1]), orphans, 1)}
        rep[col] = res
        log(f"VAL {col}: plain-rule valB={res['plain']['valB']:.5f} (hard {res['plain']['hard_valB']:.5f}) | "
            f"hard-rule valB={res['hard']['valB']:.5f} (hard {res['hard']['hard_valB']:.5f}) "
            f"train_sample={res['plain']['train_sample']:.5f} rules={res['plain']['cfg']} / {res['hard']['cfg']}")
    last = rep[cols[-1]]["hard"]
    rep["best"] = dict(last["cfg"], use=cols[-1], f05=last["valB"], hard_valB=last["hard_valB"])
    imp = dict(zip(model.feature_name(), model.feature_importance("gain").round().tolist()))
    rep["importance"] = dict(sorted(imp.items(), key=lambda x: -x[1])[:30])
    (work / "val_report_s2.json").write_text(json.dumps(rep, indent=1))
    log(f"CHOSEN {rep['best']}")


def predict_s3(work: Path, out: Path) -> None:
    import lightgbm as lgb
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    s0, s1m = (lgb.Booster(model_file=str(work / f"s2_fold{k}.txt")) for k in (0, 1))
    for m in (m0, m1, s0, s1m):
        m.best_iteration = 0
    model = lgb.Booster(model_file=str(work / "s3_model.txt"))
    rec = pl.read_parquet(work / "test" / "records.parquet",
                          columns=["entity_id", "src", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _load(work, "test").join(rec.select(pl.col("rid").alias("q"), pl.col("src").alias("qsrc")), on="q", how="left")
    d = _collective(_p1(d, m0, m1), rec)
    x = _x2(d)
    fold = d["fold"].to_numpy()
    p = np.empty(d.height, np.float32)
    for k, m in ((0, s1m), (1, s0)):  # same fold rule as in training
        idx = np.flatnonzero(fold == k)
        p[idx] = m.predict(x[idx], num_threads=NPROC)
    d = _collective_from(d.with_columns(pl.Series("p2", p)), "p2", "_r2", rec)
    d = d.with_columns(pl.Series("p3", model.predict(_x3(d), num_threads=NPROC).astype(np.float32)))
    cfg = json.loads((work / "val_report_s2.json").read_text())["best"]
    best = _best_per_query(d.with_columns(pl.col(cfg["use"]).alias("prob")))
    best.write_parquet(work / "test_best_s2.parquet")
    _write_outputs(work, out, cfg, best, d.select("q", "s1"), rec)


def predict(work: Path, out: Path) -> None:
    import lightgbm as lgb
    m0, m1 = (lgb.Booster(model_file=str(work / f"s1_fold{k}.txt")) for k in (0, 1))
    m0.best_iteration = m1.best_iteration = 0
    model = lgb.Booster(model_file=str(work / "s2_model.txt"))
    rec = pl.read_parquet(work / "test" / "records.parquet",
                          columns=["entity_id", "src", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
    d = _load(work, "test").join(rec.select(pl.col("rid").alias("q"), pl.col("src").alias("qsrc")), on="q", how="left")
    d = _collective(_p1(d, m0, m1), rec)
    d = d.with_columns(pl.Series("p2", model.predict(_x2(d), num_threads=NPROC).astype(np.float32)))
    # The decision config comes from validation; scoring can run before it exists (wait for it here).
    cfg_path = work / "val_report_s2.json"
    while not cfg_path.exists():
        log("waiting for val_report_s2.json ...")
        time.sleep(60)
    cfg = json.loads(cfg_path.read_text())["best"]
    col = "p2" if cfg["use"] == "stage2" else "p1"
    best = _best_per_query(d.with_columns(pl.col(col).alias("prob")))
    best.write_parquet(work / "test_best_s2.parquet")
    _write_outputs(work, out, cfg, best, d.select("q", "s1"), rec)


def finalize(work: Path, out: Path) -> None:
    """Decisions + output files from saved test_best_s2.parquet (no rescoring)."""
    cfg = json.loads((work / "val_report_s2.json").read_text())["best"]
    best = pl.read_parquet(work / "test_best_s2.parquet")
    cands = pl.concat([pl.read_parquet(p, columns=["q", "s1", "pA"]) for p in sorted((work / "test" / "pruned").glob("*.parquet"))])
    cands = cands.filter(pl.col("pA") >= cand_min_pa(work)).select("q", "s1")
    rec = pl.read_parquet(work / "test" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
    _write_outputs(work, out, cfg, best, cands, rec)


def _write_outputs(work: Path, out: Path, cfg: dict, best: pl.DataFrame, cands: pl.DataFrame, rec: pl.DataFrame) -> None:
    chosen = _chosen(best, cfg)
    ids = rec.select("rid", "entity_id")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))

    def agg(pairs: pl.DataFrame, c: str) -> pl.DataFrame:
        g = (pairs.join(ids.rename({"rid": "q", "entity_id": "m"}), on="q")
                  .group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(c)))
        return s1.join(g, on="s1", how="left").with_columns(pl.col(c).fill_null("")).select("source1_entity_id", c)

    out.mkdir(parents=True, exist_ok=True)
    m = agg(chosen, "matched_entity_ids")
    m.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    agg(cands, "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"stage2 predict: wrote {out} cfg={cfg} nonempty={int((m['matched_entity_ids'] != '').sum())}/{m.height}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prune", "train", "train_v5", "train_s3", "predict_s3", "evaluate", "predict", "finalize", "all", "export"])
    ap.add_argument("--src_work", type=Path)
    ap.add_argument("--orphan_w", type=float, default=3.0)
    ap.add_argument("--hard_k", type=int, default=3)
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/output_s2"))
    a = ap.parse_args()
    if a.stage in ("prune", "all"):
        prune(a.work, "train")
        prune(a.work, "test")
    if a.stage == "export":
        export_pruned(a.work, "train")
        export_pruned(a.work, "test")
    if a.stage in ("train", "all"):
        train(a.data, a.work)
    if a.stage == "evaluate":
        evaluate(a.data, a.work)
    if a.stage == "train_v5":
        train_v5(a.data, a.work, a.src_work, a.orphan_w, a.hard_k)
    if a.stage == "train_s3":
        train_s3(a.data, a.work, a.src_work, a.orphan_w, a.hard_k)
    if a.stage == "predict_s3":
        predict_s3(a.work, a.out)
    if a.stage in ("predict", "all"):
        predict(a.work, a.out)
    if a.stage == "finalize":
        finalize(a.work, a.out)
