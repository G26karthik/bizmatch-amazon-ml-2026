"""Separate decision strictness for empty-address records, validated two ways on the exact local split:
 standard  : validation as used for every production decision (19% S1 hiding, copies of hidden S1s are non-matches)
 test-like : the records that exist only because of the hiding (copies of hidden S1s, 'H') are removed, since test has
             none (empty-address records per S1: train 0.168/0.139 = test 0.1665/0.138 for US/India).
prob -> prob**g_e for records with an empty address, prob**g_a otherwise (production g_a = g_e = 1.5), then the exact
expected-F0.5 subset choice. g_e chosen on half A, reported on half B with a paired bootstrap; 'hard' counts every false
match from an unmatched record 3x (test has ~2x the decoy density).
--newp FILE: use recalibrated uncertain-zone probabilities (q, s1, pn) instead of production ones.

uv run --no-project --with-requirements requirements.txt python scripts/deep/testlike_rule.py [--newp local_data/llm/resid_val.parquet]
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import decide_pairs  # noqa: E402

SEED, VAL_MOD, HIDE = 42, 10, 19
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"),
                       (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
tp_all = pl.read_parquet("local_data/train_pairs.parquet")
hid_ids = rec.filter(pl.col("hidden")).select(pl.col("entity_id").alias("s1"))
Hset = set(rec.join(tp_all.join(hid_ids, on="s1").select(pl.col("m").alias("entity_id")), on="entity_id")["rid"].to_list())
gt = tp_all.join(hid_ids, on="s1", how="anti")
live = rec.filter(~pl.col("hidden"))
id2 = live.select("entity_id", "rid")
parent = gt.join(id2.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2.rename({"entity_id": "s1", "rid": "parent"}), on="s1").select("q", "parent")
grp = live.select("rid", "src", "entity_id", (pl.col("entity_id").hash(SEED) % VAL_MOD == 0).alias("own_val"))
lab = grp.join(parent, left_on="rid", right_on="q", how="left")
lab = lab.join(grp.select(pl.col("rid").alias("parent"), pl.col("own_val").alias("parent_val")), on="parent", how="left")
lab = lab.with_columns(pl.coalesce("parent_val", "own_val").alias("is_val"),
                       (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
S = {h: lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (h == "A")))["rid"].to_list() for h in "AB"}
par = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in par.items():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
valS1 = set(S["A"]) | set(S["B"])
empty = set(rec.filter(pl.col("empty"))["rid"].to_list())
print("labels ready; H records", len(Hset), "orphans", len(orph), flush=True)

uv = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("prob").cast(pl.Float64))
if "--newp" in sys.argv:
    nw = pl.read_parquet(sys.argv[sys.argv.index("--newp") + 1]).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "pn")
    if "--scope_empty" in sys.argv:   # recalibrated probabilities only for empty-address records
        emp_ids = set(rec.filter(pl.col("empty"))["rid"].to_list())
        nw = nw.filter(pl.col("q").is_in(list(emp_ids)))
    if "--scope_addr" in sys.argv:
        emp_ids = set(rec.filter(pl.col("empty"))["rid"].to_list())
        nw = nw.filter(~pl.col("q").is_in(list(emp_ids)))
    uv = uv.join(nw, on=["q", "s1"], how="left").with_columns(pl.coalesce("pn", "prob").cast(pl.Float64).alias("prob")).drop("pn")
    print("recalibrated probabilities joined:", nw.height, flush=True)
uq = set(uv["q"].to_list())
cert = pl.DataFrame([(q, p, 0.999) for q, p in par.items() if p in valS1 and q not in uq], schema={"q": pl.UInt32, "s1": pl.UInt32, "prob": pl.Float64}, orient="row")
best_all = pl.concat([uv.select("q", "s1", "prob"), cert]).with_columns(pl.col("q").is_in(list(empty)).alias("e"))
DUP = 0
if "--dup_noise" in sys.argv:   # test has ~2x decoy density: every noise record's uncertain pair gets a virtual twin
    OFF = 50_000_000
    nz = best_all.filter(pl.col("q").is_in(list(orph - Hset)))
    twins = nz.with_columns((pl.col("q") + OFF).cast(pl.UInt32).alias("q"))
    DUP = twins.height
    orph |= set(twins["q"].to_list())
    if "--newp" in sys.argv:   # twins of production-prob rows need the matching base rows too
        pass
    best_all = pl.concat([best_all, twins])
    print("noise twins added:", DUP, flush=True)
EMPTY_IDS = None


def decide(best, ge, ga):
    b = best.with_columns(pl.when(pl.col("e")).then(pl.col("prob") ** ge).otherwise(pl.col("prob") ** ga).alias("prob"))
    return decide_pairs(b.select("q", "s1", "prob"), 0.0)


def per_entity(best, ge, ga, s_list, drop_h):
    ch = decide(best, ge, ga)
    plain, hard = {}, {}
    for q, s in ch.select("q", "s1").iter_rows():
        if drop_h and q in Hset:
            continue
        plain.setdefault(s, set()).add(q); hard.setdefault(s, set()).add(q)
        if q in orph and q not in Hset:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
        elif q in Hset and not drop_h:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s_list]) for m, pm in (("plain", plain), ("hard", hard))}


GRID = [1.5, 1.25, 1.0, 0.8, 0.6, 0.45, 0.3]
if "--global_grid" in sys.argv:   # one strictness for every record, on the (re)calibrated table, test-like only
    best = best_all.filter(~pl.col("q").is_in(list(Hset)))
    base = {h: per_entity(best, 1.5, 1.5, S[h], True) for h in "AB"}
    res = []
    for g in (1.0, 1.25, 1.75, 2.0, 2.5):
        a = per_entity(best, g, g, S["A"], True)
        res.append((g, a["plain"].mean() - base["A"]["plain"].mean(), a["hard"].mean() - base["A"]["hard"].mean()))
        print(f"[global] half A gamma {g}: plain {res[-1][1]:+.6f} hard {res[-1][2]:+.6f}", flush=True)
    g = max(res, key=lambda r: r[1] + r[2])[0]
    bb = per_entity(best, g, g, S["B"], True)
    rng = np.random.default_rng(0)
    for m in ("plain", "hard"):
        d = bb[m] - base["B"][m]; n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        print(f"[global] half B gamma {g} vs 1.5 {m}: diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}]", flush=True)
    sys.exit(0)
if "--base_only" in sys.argv:   # compare the (re)calibrated table against production at the production rule
    uv0 = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("prob").cast(pl.Float64))
    uv0 = uv0.filter(~pl.col("q").is_in(list(Hset)))
    b0 = pl.concat([uv0.select("q", "s1", "prob"), cert]).with_columns(pl.col("q").is_in(list(empty)).alias("e"))
    if DUP:
        nz0 = b0.filter(pl.col("q").is_in(list(orph)))
        b0 = pl.concat([b0, nz0.with_columns((pl.col("q") + OFF).cast(pl.UInt32).alias("q"))])
    b1 = best_all.filter(~pl.col("q").is_in(list(Hset)))
    rng = np.random.default_rng(0)
    for h in "AB":
        x0, x1 = per_entity(b0, 1.5, 1.5, S[h], True), per_entity(b1, 1.5, 1.5, S[h], True)
        for m in ("plain", "hard"):
            d = x1[m] - x0[m]; n = len(d)
            bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
            print(f"[test-like] half {h} {m}: {x0[m].mean():.6f} -> {x1[m].mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}]", flush=True)
    sys.exit(0)
for mode in ("standard", "test-like"):
    drop_h = mode == "test-like"
    best = best_all.filter(~pl.col("q").is_in(list(Hset))) if drop_h else best_all
    base = {h: per_entity(best, 1.5, 1.5, S[h], drop_h) for h in "AB"}
    res = []
    for ge in GRID[1:]:
        a = per_entity(best, ge, 1.5, S["A"], drop_h)
        res.append((ge, a["plain"].mean() - base["A"]["plain"].mean(), a["hard"].mean() - base["A"]["hard"].mean()))
        print(f"[{mode}] half A g_e {ge}: plain {res[-1][1]:+.6f} hard {res[-1][2]:+.6f}", flush=True)
    ge = max(res, key=lambda r: r[1] + r[2])[0]
    bb = per_entity(best, ge, 1.5, S["B"], drop_h)
    rng = np.random.default_rng(0)
    for m in ("plain", "hard"):
        d = bb[m] - base["B"][m]; n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        print(f"[{mode}] half B g_e {ge} {m}: {base['B'][m].mean():.6f} -> {bb[m].mean():.6f} diff {d.mean():+.6f} "
              f"CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(<=0) {np.mean(bs <= 0):.3f}", flush=True)
