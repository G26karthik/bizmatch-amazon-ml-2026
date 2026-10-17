"""Average per-pair stage-3 probabilities of several runs (same candidate set) in logit space.

Each run directory must contain scores_val.parquet (q, s1, p3, y) and scores_test.parquet (q, s1, p3) from
`python -m ber.blend dump`. Optional cross-encoder blending is applied on top via ber.blend when ce is given.
The decision rule is chosen on validation half A (plain and decoy-hard); half B is reported.

python -m ber.ensemble --data /work/data --runs /work/run6 /work/run7 --out /work/output_ens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ber.blend import _logit
from ber.pipeline import SEED, _best_per_query, labels, log
from ber.stage2 import _chosen, _macro_hard, _write_outputs


def _mean_logit(frames: list[pl.DataFrame]) -> pl.DataFrame:
    base = frames[0].select("q", "s1", *[c for c in ("y",) if c in frames[0].columns])
    z = np.zeros(base.height)
    for f in frames:
        g = base.select("q", "s1").join(f.select("q", "s1", "p3"), on=["q", "s1"], how="left")
        missing = g["p3"].null_count()
        if missing or f.height != base.height:  # members must score exactly the same pairs (else silent bias)
            raise ValueError(f"ensemble members disagree on pairs: {missing} missing, {f.height} vs {base.height}")
        z += _logit(g["p3"].to_numpy())
    z /= len(frames)
    return base.with_columns(pl.Series("prob", (1 / (1 + np.exp(-z))).astype(np.float32)))


def run(data: Path, runs: list[Path], out: Path | None, hard_k: int = 3, save_dir: Path | None = None) -> None:
    lab = labels(data, runs[0]).with_columns(
        (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    s1_all = lab.filter(pl.col("src") == 1)
    s1_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    orphans = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    grid = ([{"mode": "thr", "thr": round(float(t), 2)} for t in np.arange(0.5, 0.96, 0.05)] +
            [{"mode": "efopt", "miss": 0.0, "gamma": g} for g in (0.8, 1.0, 1.25, 1.5, 2.0)])
    vals = [pl.read_parquet(r / "scores_val.parquet") for r in runs]
    rep: dict = {"runs": [str(r) for r in runs]}
    for name, frames in [(r.name, [v]) for r, v in zip(runs, vals)] + [("ensemble", vals)]:
        best = _best_per_query(_mean_logit(frames))
        res = {}
        for rule, k in (("plain", 1), ("hard", hard_k)):
            top = max(((_macro_hard(s1_a, truth, _chosen(best, c), orphans, k), c) for c in grid), key=lambda x: x[0])
            ch = _chosen(best, top[1])
            res[rule] = {"cfg": top[1], "valB": _macro_hard(s1_b, truth, ch, orphans, 1),
                         "hard_valB": _macro_hard(s1_b, truth, ch, orphans, hard_k)}
        rep[name] = res
        log(f"VAL {name}: plain-rule valB={res['plain']['valB']:.5f} (hard {res['plain']['hard_valB']:.5f}) | "
            f"hard-rule valB={res['hard']['valB']:.5f} (hard {res['hard']['hard_valB']:.5f})")
    (runs[0].parent / "val_report_ensemble.json").write_text(json.dumps(rep, indent=1))
    if save_dir is not None:  # write the averaged probabilities as a pseudo-run for ber.blend (CE on top)
        import shutil
        save_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("train", "test"):
            link = save_dir / sub
            if not link.exists():
                link.symlink_to((runs[0] / sub).resolve(), target_is_directory=True)
        shutil.copy(runs[0] / "config.json", save_dir / "config.json")
        _mean_logit(vals).rename({"prob": "p3"}).write_parquet(save_dir / "scores_val.parquet")
        _mean_logit([pl.read_parquet(r / "scores_test.parquet") for r in runs]).rename({"prob": "p3"})             .write_parquet(save_dir / "scores_test.parquet")
        log(f"saved ensemble scores to {save_dir}")
    if out is not None:
        t = _mean_logit([pl.read_parquet(r / "scores_test.parquet") for r in runs])
        rec = pl.read_parquet(runs[0] / "test" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
        cfg = rep["ensemble"]["hard"]["cfg"]
        _write_outputs(runs[0], out, cfg, _best_per_query(t), t.select("q", "s1"), rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--save_dir", type=Path)
    a = ap.parse_args()
    run(a.data, a.runs, a.out, save_dir=a.save_dir)
