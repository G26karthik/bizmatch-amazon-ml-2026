"""Incremental value of the legal-form collision resolver over production, on validation (local, no AWS).
Production decisions are reconstructed from local_data/llm/unc_val.parquet (run28_7q best-S1 pairs with
0.02 < p < 0.98) plus every other true copy of a validation S1 assumed accepted (p 0.999): the production rule
(efopt, gamma 1.5) is re-run on that. The resolver acts only on uncertain empty-address records whose name key matches
2+ (non-hidden) S1s of the country: the candidate with the record's legal form (unique), else the unique drop/add one,
gets the record with probability = the rule precision measured on half A (by country x candidate count); the decision
rule is re-run and half B is scored (paired bootstrap, plain and hard).

uv run --no-project --with polars==1.44.2 --with numpy --with numba --with lightgbm --with rapidfuzz --with scikit-learn python scripts/coll_val.py
"""
import re
import sys
import unicodedata

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

SEED, VAL_MOD, HIDE = 42, 10, 19
CFG = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
CANON = {"ltd": "limited", "limited": "limited", "pvt": "private", "private": "private", "inc": "inc", "incorporated": "inc",
         "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "llc": "llc", "lp": "lp", "llp": "llp", "pc": "pc",
         "pllc": "pllc", "plc": "plc", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
         "snc": "snc", "ei": "ei"}
STOP = {"the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def toks(s):
    t = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.findall(r"[a-z0-9]+", t)


def key(s):
    return " ".join(sorted(set(toks(s)) - set(CANON) - STOP))


def legal(s):
    return frozenset(CANON[t] for t in toks(s) if t in CANON)


# ---- local labels (same as ber.pipeline.labels with hide_pct 19)
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]) \
        .with_row_index("rid")
hidden = (pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)
rec = rec.with_columns(hidden.alias("hidden"))
gt = pl.read_parquet("local_data/train_pairs.parquet")
hid_ids = rec.filter(pl.col("hidden")).select(pl.col("entity_id").alias("s1"))
gt = gt.join(hid_ids, on="s1", how="anti")
live = rec.filter(~pl.col("hidden"))
id2 = live.select("entity_id", "rid")
parent = gt.join(id2.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2.rename({"entity_id": "s1", "rid": "parent"}), on="s1").select("q", "parent")
grp = live.select("rid", "src", "entity_id", (pl.col("entity_id").hash(SEED) % VAL_MOD == 0).alias("own_val"))
lab = grp.join(parent, left_on="rid", right_on="q", how="left")
lab = lab.join(grp.select(pl.col("rid").alias("parent"), pl.col("own_val").alias("parent_val")), on="parent", how="left")
lab = lab.with_columns(pl.coalesce("parent_val", "own_val").alias("is_val"),
                       (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
S = {h: lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (h == "A")))["rid"].to_list() for h in "AB"}
print("val S1 half A", len(S["A"]), "half B", len(S["B"]), "(expect 89,043 for B)", flush=True)
par = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in par.items():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
valS1 = set(S["A"]) | set(S["B"])
half = dict(lab.filter(pl.col("src") == 1).select("rid", "half_a").iter_rows())

# ---- approximate production best-S1 table on validation
uv = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
uq = set(uv["q"].to_list())
cert = [(q, p, 0.999) for q, p in par.items() if p in valS1 and q not in uq]
best = pl.concat([uv.select("q", "s1", pl.col("prob").cast(pl.Float64)),
                  pl.DataFrame(cert, schema={"q": pl.UInt32, "s1": pl.UInt32, "prob": pl.Float64}, orient="row")])
print("approx best: uncertain", uv.height, "certain", len(cert), flush=True)

# ---- collision resolver candidates among uncertain empty-address records
nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); cty = rec["country"].to_list(); srcl = rec["src"].to_list()
s1live = live.filter(pl.col("src") == 1).select("rid", "business_name", "country")
kidx = {}
for r, n, c in s1live.iter_rows():
    kidx.setdefault((c, key(n)), []).append(r)
rows = []
for q, s, p in uv.iter_rows():
    if ad[q] and ad[q].strip():
        continue
    cands = kidx.get((cty[q], key(nm[q])), [])
    if len(cands) < 2:
        continue
    lr = legal(nm[q])
    rel = []
    for c in cands:
        ls = legal(nm[c])
        rel.append("same" if ls == lr else ("drop" if lr < ls else ("add" if ls < lr else "swap")))
    same = [c for c, r in zip(cands, rel) if r == "same"]
    pick = same[0] if len(same) == 1 else None
    if pick is None and not same:
        da = [c for c, r in zip(cands, rel) if r in ("drop", "add")]
        pick = da[0] if len(da) == 1 else None
    if pick is None:
        continue
    rows.append((q, s, p, pick, len(cands), cty[q], half.get(pick, False)))
cz = pl.DataFrame(rows, schema=["q", "s1_best", "p_best", "pick", "n", "country", "hA"], orient="row")
cz = cz.with_columns(pl.Series("y_pick", [par.get(q) == k for q, k in zip(cz["q"].to_list(), cz["pick"].to_list())]),
                     pl.Series("y_best", [par.get(q) == k for q, k in zip(cz["q"].to_list(), cz["s1_best"].to_list())]),
                     pl.col("n").clip(2, 6).alias("nb"))
print("resolver fires on uncertain val records:", cz.height, " pick == best S1:", (cz["pick"] == cz["s1_best"]).mean(), flush=True)
print(cz.group_by("country", "nb").agg(pl.len(), pl.col("y_pick").mean().round(3).alias("prec_pick"), pl.col("y_best").mean().round(3).alias("prec_best"),
      (pl.col("pick") == pl.col("s1_best")).mean().round(3).alias("agree")).sort("country", "nb"))
# precision table from half A only (records whose picked S1 is in half A)
ptab = cz.filter(pl.col("hA")).group_by("country", "nb", (pl.col("pick") == pl.col("s1_best")).alias("agree")).agg(pl.col("y_pick").mean().alias("pp"), pl.len().alias("np"))
cz = cz.with_columns((pl.col("pick") == pl.col("s1_best")).alias("agree")).join(ptab, on=["country", "nb", "agree"], how="left")


def per_entity(bst, s_list):
    ch = _chosen(bst, CFG)
    plain, hard = {}, {}
    for q, s in ch.select("q", "s1").iter_rows():
        plain.setdefault(s, set()).add(q)
        hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s_list]) for m, pm in (("plain", plain), ("hard", hard))}


def variant(min_p, only_disagree=False):
    z = cz.filter(pl.col("pp") >= min_p)
    if only_disagree:
        z = z.filter(~pl.col("agree"))
    mod = z.select(pl.col("q").cast(pl.UInt32), pl.col("pick").cast(pl.UInt32).alias("s1"), pl.col("pp").cast(pl.Float64).alias("prob"))
    return pl.concat([best.join(mod.select("q"), on="q", how="anti"), mod]), z.height


base = {h: per_entity(best, S[h]) for h in "AB"}
print("approx production half A plain %.6f  half B plain %.6f hard %.6f (real half B 0.991316 / 0.990823)" %
      (base["A"]["plain"].mean(), base["B"]["plain"].mean(), base["B"]["hard"].mean()), flush=True)
rng = np.random.default_rng(0)
res = []
for mp in (0.0, 0.6, 0.7, 0.8, 0.85, 0.9):
    for od in (False, True):
        b2, n = variant(mp, od)
        a = per_entity(b2, S["A"])
        res.append((mp, od, n, a["plain"].mean() - base["A"]["plain"].mean(), a["hard"].mean() - base["A"]["hard"].mean()))
        print(f"half A  min_prec {mp:.2f} only_disagree {od}: records {n}  plain {res[-1][3]:+.6f} hard {res[-1][4]:+.6f}", flush=True)
bestv = max(res, key=lambda r: r[3] + r[4])
print("half-A pick:", bestv, flush=True)
b2, n = variant(bestv[0], bestv[1])
bb = per_entity(b2, S["B"])
for m in ("plain", "hard"):
    d = bb[m] - base["B"][m]; nn = len(d)
    bs = np.array([d[rng.integers(0, nn, nn)].mean() for _ in range(2000)])
    print(f"half B {m}: {base['B'][m].mean():.6f} -> {bb[m].mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(<=0) {np.mean(bs <= 0):.3f}")
ptab.write_parquet("local_data/work/coll_ptab.parquet")
