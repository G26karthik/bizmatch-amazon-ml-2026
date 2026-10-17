"""E00 band pilot: include the Stage-A band 0.001 <= pA < 0.01 with the run28 stack models (reused) in /work/run28b.

phases (run in order):
  sanity   reproduce run28 p3 (scores_val) on an exact 2-hop subgraph of 1500 validation S1s with cand_min_pA 0.01
  val      p3 for all training pairs with pA >= 0.001 -> /work/run28b/scores_val.parquet (eval queries only)
  blend    ber.blend gbm with the 7 run28_7q cross-encoders + raw feats, tag run28b_7q (test part only if scores_test exists)
  compare  paired bootstrap vs run28_7q on half B at efopt gamma 1.5 (plain / hard) + added/removed pairs by truth class
  test     p3 for all test pairs -> scores_test.parquet, then blend again (-> test_best_run28b_7q), band pairs in decisions
python scripts/e00_band_pilot.py sanity val blend compare      (NT threads, default 6)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
NT = int(os.environ.get("NT", "6"))
import ber.blend as B  # noqa: E402
import ber.pipeline as P  # noqa: E402
import ber.stage2 as S2  # noqa: E402
import rapidfuzz.process as RP  # noqa: E402

for _m in (P, S2, B):
    _m.NPROC = NT
_cpdist = RP.cpdist


def _cp(*a, **k):
    k["workers"] = NT
    return _cpdist(*a, **k)


RP.cpdist = _cp  # stage2._siblings imports cpdist at call time
T0 = time.time()
work = Path("/work/run28b")
data = Path("/work/data")
TAG, REF = "run28b_7q", "run28_7q"
CE = [Path(f"/work/{x}") for x in ("ce_wide", "ce_wide_base", "ce_wide_base2", "ce_large", "ce_bgefull", "ce_e5lfull", "ce_qwen3r")]
cfg = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def qlab():
    lab = P.labels(data, work)
    return lab, lab.select(pl.col("rid").alias("q"), "parent", "is_val", pl.col("src").alias("qsrc"))


def load(cut: float) -> pl.DataFrame:
    S2.cand_min_pa = lambda w: cut
    return S2._load(work, "train")


def sanity():
    lab, ql = qlab()
    d = load(0.01).join(ql, on="q", how="left")
    val_s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val"))["rid"]
    A = val_s1.sample(1500, seed=5).cast(d["s1"].dtype)
    T = d.filter(pl.col("s1").is_in(A.implode())).select("q", "s1")
    n1 = d.filter(pl.col("q").is_in(T["q"].unique().implode()) | pl.col("s1").is_in(A.implode()))
    n2 = d.filter(pl.col("q").is_in(n1["q"].unique().implode()) | pl.col("s1").is_in(n1["s1"].unique().implode()))
    log(f"sanity subgraph: target pairs {T.height}, hop1 {n1.height}, hop2 {n2.height}")
    del d
    p = B._p3(work, "train", n2).join(T, on=["q", "s1"], how="semi").select("q", "s1", "p3")
    ref = pl.read_parquet("/work/run28/scores_val.parquet", columns=["q", "s1", "p3"]).rename({"p3": "ref"}).with_columns(
        pl.col("q").cast(p["q"].dtype), pl.col("s1").cast(p["s1"].dtype))
    c = p.join(ref, on=["q", "s1"], how="inner").with_columns((pl.col("p3") - pl.col("ref")).abs().alias("ad"))
    mx, q999 = c["ad"].max(), c["ad"].quantile(0.999)
    log(f"sanity: compared {c.height}/{p.height} pairs; |p3 - run28 p3| mean {c['ad'].mean():.2e} p99.9 {q999:.2e} max {mx:.2e}")
    if c.height < 0.99 * p.height or mx > 1e-3:
        log("SANITY FAILED -> no-go")
        sys.exit(2)
    log("SANITY OK")


def sanity2():
    """Is the sanity gap pipeline noise? Same subgraph scored twice (original order / shuffled rows) vs run28 p3."""
    lab, ql = qlab()
    d = load(0.01).join(ql, on="q", how="left")
    val_s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val"))["rid"]
    A = val_s1.sample(1500, seed=5).cast(d["s1"].dtype)
    T = d.filter(pl.col("s1").is_in(A.implode())).select("q", "s1")
    n1 = d.filter(pl.col("q").is_in(T["q"].unique().implode()) | pl.col("s1").is_in(A.implode()))
    n2 = d.filter(pl.col("q").is_in(n1["q"].unique().implode()) | pl.col("s1").is_in(n1["s1"].unique().implode()))
    del d
    ref = pl.read_parquet("/work/run28/scores_val.parquet", columns=["q", "s1", "p3"]).rename({"p3": "ref"}).with_columns(
        pl.col("q").cast(n2["q"].dtype), pl.col("s1").cast(n2["s1"].dtype))
    runs = {}
    for name, fr in (("a", n2), ("b", n2.sample(fraction=1.0, shuffle=True, seed=11))):
        runs[name] = B._p3(work, "train", fr).join(T, on=["q", "s1"], how="semi").select("q", "s1", pl.col("p3").alias(name))
    c = runs["a"].join(runs["b"], on=["q", "s1"]).join(ref, on=["q", "s1"])
    for x, y in (("a", "ref"), ("b", "ref"), ("a", "b")):
        ad = (c[x] - c[y]).abs()
        log(f"|{x} - {y}|: mean {ad.mean():.2e} p99 {ad.quantile(0.99):.2e} p99.9 {ad.quantile(0.999):.2e} max {ad.max():.2e} "
            f"share>1e-4 {(ad > 1e-4).mean():.4f}")


def sanity3():
    """Exact reproduction check on a closed-enough subgraph: expand (all pairs of touched records / S1s) HOPS times."""
    hops, nA = int(os.environ.get("HOPS", "5")), int(os.environ.get("NA", "300"))
    lab, ql = qlab()
    d = load(0.01).join(ql, on="q", how="left")
    val_s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val"))["rid"]
    A = val_s1.sample(nA, seed=5).cast(d["s1"].dtype)
    T = d.filter(pl.col("s1").is_in(A.implode())).select("q", "s1")
    F = T
    sizes = []
    for _ in range(hops):
        F = d.filter(pl.col("q").is_in(F["q"].unique().implode()) | pl.col("s1").is_in(F["s1"].unique().implode()))
        sizes.append(F.height)
    log(f"sanity3: target pairs {T.height}; hop sizes {sizes}")
    del d
    ref = pl.read_parquet("/work/run28/scores_val.parquet", columns=["q", "s1", "p3"]).rename({"p3": "ref"}).with_columns(
        pl.col("q").cast(F["q"].dtype), pl.col("s1").cast(F["s1"].dtype))
    c = B._p3(work, "train", F).join(T, on=["q", "s1"], how="semi").select("q", "s1", "p3").join(ref, on=["q", "s1"])
    ad = (c["p3"] - c["ref"]).abs()
    log(f"sanity3 |p3 - run28|: n {c.height} mean {ad.mean():.2e} p99 {ad.quantile(0.99):.2e} p99.9 {ad.quantile(0.999):.2e} "
        f"max {ad.max():.2e} share>1e-4 {(ad > 1e-4).mean():.4f}")


def val():
    lab, ql = qlab()
    d = load(0.001).join(ql, on="q", how="left")
    s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))["rid"].cast(d["s1"].dtype)
    qs = d.filter(pl.col("s1").is_in(s1v.implode()))["q"].unique()
    log(f"train pairs pA>=0.001: {d.height} (band {d.filter(pl.col('pA') < 0.01).height}); eval queries {qs.len()}")
    d = B._p3(work, "train", d)
    d = d.filter(pl.col("q").is_in(qs.implode()))
    d.select("q", "s1", "p3", "is_val", (pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"), "pA") \
     .write_parquet(work / "scores_val.parquet")
    log(f"wrote scores_val: {d.height} pairs")


def blend():
    try:
        B.blend(data, work, CE, work / "out", model="gbm", tag=TAG, raw=Path("/work/rawf"))
    except Exception as ex:  # noqa: BLE001 - val-only mode: the test part needs scores_test.parquet
        if (work / "scores_test.parquet").exists():
            raise
        log(f"blend: validation part done, test part skipped ({type(ex).__name__})")
    log("blend done")


def compare():
    from ber.metric import entity_f05
    from ber.pipeline import SEED, hidden_s1
    lab = P.labels(data, work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    s1_b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    sb = set(s1_b)
    parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
    truth = {}
    for q, p in parent.items():
        truth.setdefault(p, set()).add(q)
    orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    gt = pl.read_parquet("/work/data/train_pairs.parquet")
    H = set(lab.join(gt.filter(hidden_s1(pl.col("s1"), 19)).select(pl.col("m").alias("entity_id")), on="entity_id", how="semi")["rid"].to_list())
    res, chs = {}, {}
    for name, path in ((REF, Path("/work/run28") / f"val_best_{REF}.parquet"), (TAG, work / f"val_best_{TAG}.parquet")):
        ch = S2._chosen(pl.read_parquet(path).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)), cfg)
        chs[name] = set((q, s) for q, s in ch.iter_rows() if s in sb)
        plain, hard = {}, {}
        for q, s in ch.iter_rows():
            plain.setdefault(s, set()).add(q)
            hard.setdefault(s, set()).add(q)
            if q in orph:
                hard[s] |= {("d", q, 1), ("d", q, 2)}
        res[name] = {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1_b]) for m, pm in (("plain", plain), ("hard", hard))}
        log(name, {m: round(float(v.mean()), 6) for m, v in res[name].items()})
    rng = np.random.default_rng(0)
    for m in ("plain", "hard"):
        d = res[TAG][m] - res[REF][m]
        n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        log(f"half B {m}: diff={d.mean():+.6f} 95%CI=[{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] "
            f"P(diff<=0)={np.mean(bs <= 0):.3f} better={np.sum(d > 0)} worse={np.sum(d < 0)}")
    sv = pl.read_parquet(work / "scores_val.parquet", columns=["q", "s1", "pA"])
    band = set((q, s) for q, s in sv.filter(pl.col("pA") < 0.01).select("q", "s1").iter_rows())

    def cls(q, s):
        p = parent.get(q)
        return "true" if p == s else "other_S1_copy" if p is not None else "hiddenS1_copy" if q in H else "unmatched"

    for lbl, st in (("added", chs[TAG] - chs[REF]), ("removed", chs[REF] - chs[TAG])):
        c = {}
        for q, s in st:
            k = cls(q, s) + ("|band" if (q, s) in band else "")
            c[k] = c.get(k, 0) + 1
        log(f"half B pairs {lbl}: {len(st)} {dict(sorted(c.items()))}; S1s {len({s for _, s in st})}")
    aff = {s for _, s in chs[TAG] ^ chs[REF]}
    log(f"half B affected S1s: {len(aff)}")


def test():
    rec = pl.read_parquet(work / "test" / "records.parquet", columns=["src"]).with_row_index("rid")
    S2.cand_min_pa = lambda w: 0.001
    t = S2._load(work, "test").join(rec.select(pl.col("rid").alias("q"), pl.col("src").alias("qsrc")), on="q", how="left")
    log(f"test pairs pA>=0.001: {t.height} (band {t.filter(pl.col('pA') < 0.01).height})")
    B._p3(work, "test", t).select("q", "s1", "p3").write_parquet(work / "scores_test.parquet")
    del t
    log("wrote scores_test")
    blend()
    trec = pl.read_parquet(work / "test" / "records.parquet", columns=["country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    pa = pl.concat([pl.read_parquet(f, columns=["q", "s1", "pA"]) for f in sorted((work / "test" / "pruned").glob("*.parquet"))]) \
           .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    out = {}
    for name, path in ((REF, Path("/work/run28") / f"test_best_{REF}.parquet"), (TAG, work / f"test_best_{TAG}.parquet")):
        ch = S2._chosen(pl.read_parquet(path).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)), cfg)
        out[name] = ch.join(trec.select(pl.col("rid").alias("q"), "country"), on="q").filter(pl.col("country").is_in(["US", "India"])) \
                      .join(pa, on=["q", "s1"], how="left")
    new, old = out[TAG], out[REF]
    add = new.join(old.select("q", "s1"), on=["q", "s1"], how="anti")
    rem = old.join(new.select("q", "s1"), on=["q", "s1"], how="anti")
    log(f"test US/India production-cfg decisions: {REF} {old.height} pairs, {TAG} {new.height}; added {add.height} "
        f"(band {add.filter(pl.col('pA') < 0.01).height}), removed {rem.height}; band pairs in {TAG} decisions "
        f"{new.filter(pl.col('pA') < 0.01).height}")
    log(f"test_best: {work / f'test_best_{TAG}.parquet'}")


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        log(f"=== phase {ph}")
        {"sanity": sanity, "sanity2": sanity2, "sanity3": sanity3, "val": val, "blend": blend, "compare": compare, "test": test}[ph]()
