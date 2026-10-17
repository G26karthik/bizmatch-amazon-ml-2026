"""Decoder experiments (E09/E10): shared cache, generalised expected-F0.5 decoder, vectorised per-entity metric.

Cache (/work/agent_dec): val.parquet (one row per validation record: 7q argmax pair + competitor features from
run28 scores_val p3), s1.parquet (validation S1s: half, true copy count, other-candidate mass), truth/orphans as in
scripts/eval_judge.py (same labels, halves, truth, orphan set).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

sys.path.insert(0, "src")
from ber.pipeline import SEED, labels  # noqa: E402

WORK = Path("/work/run28")
OUT = Path("/work/agent_dec")
TAG = "run28_7q"
CAP = {2: 5, 3: 6}


def comp_feats(best: pl.DataFrame, sc: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """best (q, s1, prob) argmax pairs; sc (q, s1, p3) all candidate pairs. Returns q-level competitor features and
    s1-level mass of records that list the S1 as a non-argmax candidate."""
    b = best.select("q", pl.col("s1").alias("bs1"))
    x = sc.select("q", "s1", "p3").join(b, on="q", how="inner")
    oth = x.filter(pl.col("s1") != pl.col("bs1"))
    self_p3 = x.filter(pl.col("s1") == pl.col("bs1")).select("q", pl.col("p3").alias("p3self"))
    q2 = (oth.sort("p3", descending=True).group_by("q", maintain_order=True)
             .agg(pl.col("p3").first().alias("c_max"), pl.col("s1").first().alias("s2nd"), pl.col("p3").sum().alias("c_sum"),
                  pl.len().alias("n_other")))
    qf = best.join(self_p3, on="q", how="left").join(q2, on="q", how="left").with_columns(
        pl.col("c_max").fill_null(0.0), pl.col("c_sum").fill_null(0.0), pl.col("n_other").fill_null(0).cast(pl.Int32))
    sf = oth.group_by("s1").agg(pl.col("p3").sum().alias("om_sum"), pl.col("p3").max().alias("om_max"),
                                (pl.col("p3") > 0.1).sum().alias("om_n10"), pl.len().alias("om_cnt"))
    return qf, sf


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = labels(Path("/work/data"), WORK).with_columns(
        (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val")).select(pl.col("rid").cast(pl.UInt32).alias("s1"), "half_a")
    par = lab.filter(pl.col("parent").is_not_null()).select(pl.col("rid").cast(pl.UInt32).alias("q"),
                                                            pl.col("parent").cast(pl.UInt32))
    orph = lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1)).select(pl.col("rid").cast(pl.UInt32).alias("q"))
    qsrc = lab.select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("src").cast(pl.Int8).alias("qsrc"))
    best = pl.read_parquet(WORK / f"val_best_{TAG}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    sc = pl.read_parquet(WORK / "scores_val.parquet", columns=["q", "s1", "p3"]).with_columns(
        pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    qf, sf = comp_feats(best, sc)
    tpar = sc.join(par, on="q", how="inner").filter(pl.col("s1") == pl.col("parent")).select("q", pl.col("p3").alias("p3true"))
    v = (qf.join(par, on="q", how="left").join(qsrc, on="q", how="left")
           .join(orph.with_columns(pl.lit(True).alias("orph")), on="q", how="left").join(tpar, on="q", how="left")
           .with_columns(pl.col("orph").fill_null(False), (pl.col("parent") == pl.col("s1")).fill_null(False).alias("y")))
    v.write_parquet(OUT / "val.parquet")
    T = par.group_by("parent").agg(pl.len().alias("T")).rename({"parent": "s1"})
    tb = v.filter(pl.col("y")).group_by("s1").agg(pl.len().alias("T_best"))
    nb = v.group_by("s1").agg(pl.len().alias("n_best"), (pl.col("prob") > 0.5).sum().alias("n_best50"),
                              pl.col("prob").sum().alias("best_mass"))
    s1 = (s1.join(T, on="s1", how="left").join(tb, on="s1", how="left").join(nb, on="s1", how="left").join(sf, on="s1", how="left")
            .with_columns(pl.col(c).fill_null(0) for c in ("T", "T_best", "n_best", "n_best50", "best_mass", "om_sum", "om_max",
                                                            "om_n10", "om_cnt")))
    s1.write_parquet(OUT / "s1.parquet")
    print("cache built", v.shape, s1.shape, flush=True)


def load():
    v = pl.read_parquet(OUT / "val.parquet")
    s1 = pl.read_parquet(OUT / "s1.parquet")
    return v, s1


# ---------------------------------------------------------------- decoder
@njit(cache=True)
def _pb(p):
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for i in range(len(p)):
        pi = p[i]
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1 - pi) + d[j - 1] * pi
        d[0] *= 1 - pi
    return d


@njit(cache=True)
def _best_k(p, miss, pois, mech, e0):
    """Top-k prefix maximising E[F0.5]. miss: expected true copies outside the list; pois=0 -> deterministic extra FN
    (production ber.decide), pois=1 -> Poisson(miss) number of extra FN. mech: 0 both effects of miss, 1 only the
    empty-prediction factor exp(-miss), 2 only the extra FN in non-empty predictions. e0: extra multiplier on the
    value of the empty prediction (1.0 = exact)."""
    n = len(p)
    mfn = 0.0 if mech == 1 else miss
    if pois and mfn > 0:
        M = int(mfn + 6.0 * np.sqrt(mfn) + 3)
        w = np.zeros(M + 1)
        w[0] = np.exp(-mfn)
        for m in range(1, M + 1):
            w[m] = w[m - 1] * mfn / m
        ms = np.arange(M + 1).astype(np.float64)
    else:
        w = np.ones(1)
        ms = np.array([mfn])
    p0 = 1.0 if mech == 2 else np.exp(-miss)
    best_k, best_v = 0, _pb(p)[0] * p0 * e0
    for k in range(1, n + 1):
        tp = _pb(p[:k])
        fn = _pb(p[k:])
        v = 0.0
        for a in range(1, k + 1):
            if tp[a] == 0.0:
                continue
            inner = 0.0
            for b in range(n - k + 1):
                if fn[b] == 0.0:
                    continue
                for mi in range(len(w)):
                    inner += fn[b] * w[mi] * 1.25 * a / (1.25 * a + (k - a) + 0.25 * (b + ms[mi]))
            v += tp[a] * inner
        if v > best_v:
            best_v, best_k = v, k
    return best_k, best_v


@njit(cache=True)
def _select(starts, p, miss, pois, mech, e0):
    keep = np.zeros(len(p), dtype=np.bool_)
    ev = np.zeros(len(starts) - 1)
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        k, val = _best_k(p[s:e], miss[g], pois, mech, e0[g])
        keep[s:s + k] = True
        ev[g] = val
    return keep, ev


def decide(best: pl.DataFrame, pcol: str = "prob", gamma: float = 1.5, miss=0.0, pois: int = 0, floor: float = 0.02,
           caps: bool = False, gamma_col: str | None = None, mech: int = 0, e0: float = 1.0) -> pl.DataFrame:
    """Generalised decide_pairs. miss: float or name of an s1-level column already on `best` (per-S1 value).
    caps: drop candidates beyond the 5 S2 / 6 S3 best of each S1 before selection. Returns (q, s1, prob) chosen."""
    g = pl.col(gamma_col) if gamma_col else pl.lit(gamma)
    b = best.with_columns(pl.col(pcol).cast(pl.Float64).pow(g).alias("_p")).filter(pl.col("_p") >= floor)
    b = b.sort(["s1", "_p"], descending=[False, True])
    if caps:
        b = b.with_columns(pl.int_range(pl.len()).over(["s1", "qsrc"]).alias("_r")).filter(
            pl.when(pl.col("qsrc") == 2).then(pl.col("_r") < CAP[2]).otherwise(pl.col("_r") < CAP[3]))
    if b.height == 0:
        return b.select("q", "s1", pl.col(pcol).alias("prob"))
    s1a = b["s1"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(s1a)) + 1, [b.height]]).astype(np.int64)
    if isinstance(miss, str):
        mg = b[miss].to_numpy()[starts[:-1]].astype(np.float64)
    else:
        mg = np.full(len(starts) - 1, float(miss))
    if isinstance(e0, str):  # "raw": value of the empty prediction from the un-gamma'd probabilities
        lr = np.log1p(-np.clip(b[pcol].to_numpy().astype(np.float64), 0, 1 - 1e-12))
        lg = np.log1p(-np.clip(b["_p"].to_numpy(), 0, 1 - 1e-12))
        e0a = np.exp(np.add.reduceat(lr - lg, starts[:-1]))
    else:
        e0a = np.full(len(starts) - 1, float(e0))
    keep, _ = _select(starts, b["_p"].to_numpy().astype(np.float64), mg, int(pois), int(mech), e0a)
    return b.filter(pl.Series(keep)).select("q", "s1", pl.col(pcol).alias("prob"))


# ---------------------------------------------------------------- metric
def per_entity(chosen: pl.DataFrame, v: pl.DataFrame, s1: pl.DataFrame, half: str) -> dict:
    """Per-entity plain and hard F0.5 over validation S1s of one half (hard: false matches of unmatched records x3),
    identical to scripts/eval_judge.py per_entity (verified on the baseline)."""
    ss = s1.filter(pl.col("half_a") == (half == "A")).select("s1", "T")
    c = chosen.select("q", "s1").join(v.select("q", "parent", "orph"), on="q", how="left")
    agg = c.group_by("s1").agg((pl.col("parent") == pl.col("s1")).fill_null(False).sum().alias("tp"), pl.len().alias("k"),
                               pl.col("orph").sum().alias("fo"))
    x = ss.join(agg, on="s1", how="left").with_columns(pl.col("tp").fill_null(0), pl.col("k").fill_null(0), pl.col("fo").fill_null(0))
    T, tp, k, fo = (x[c].to_numpy().astype(np.float64) for c in ("T", "tp", "k", "fo"))
    fp = k - tp
    out = {}
    for m, fpm in (("plain", fp), ("hard", fp + 2 * fo)):
        den = 1.25 * tp + fpm + 0.25 * (T - tp)
        f = np.where(T == 0, (k == 0).astype(np.float64), np.divide(1.25 * tp, den, out=np.zeros_like(den), where=den > 0))
        out[m] = f
    out["s1"] = x["s1"].to_numpy()
    return out


def score(chosen, v, s1, half):
    r = per_entity(chosen, v, s1, half)
    return float(r["plain"].mean()), float(r["hard"].mean())


def boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    d = b - a
    N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)])
    return (f"{a.mean():.6f} -> {b.mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] "
            f"P(<=0) {np.mean(bs <= 0):.3f} better {int((d > 0).sum())} worse {int((d < 0).sum())}")


if __name__ == "__main__":
    if "build" in sys.argv:
        build()
