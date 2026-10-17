"""Per-country strictness (gamma) for the production decision, tuned on half A, one look at half B (local reconstruction as coll_val.py).

uv run --no-project --with-requirements requirements.txt python scripts/gamma_country.py
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


def per_entity(bst, s_list, cfg):
    ch = _chosen(bst, cfg)
    plain, hard = {}, {}
    for q, s in ch.select("q", "s1").iter_rows():
        plain.setdefault(s, set()).add(q)
        hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s_list]) for m, pm in (("plain", plain), ("hard", hard))}


cty = dict(zip(rec.filter(pl.col("src") == 1)["rid"].to_list(), rec.filter(pl.col("src") == 1)["country"].to_list()))
bc = best.with_columns(pl.Series("c", [cty.get(s, "") for s in best["s1"].to_list()]))


def run(gus, gin, s_list):
    parts = []
    for c, gm in (("US", gus), ("India", gin)):
        b = bc.filter(pl.col("c") == c).select("q", "s1", "prob")
        parts.append(_chosen(b, {"mode": "efopt", "miss": 0.0, "gamma": gm}).select("q", "s1"))
    ch = pl.concat(parts)
    plain, hard = {}, {}
    for q, s in ch.iter_rows():
        plain.setdefault(s, set()).add(q); hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s_list]) for m, pm in (("plain", plain), ("hard", hard))}


base = {h: run(1.5, 1.5, S[h]) for h in "AB"}
print("baseline half A %.6f  half B %.6f" % (base["A"]["plain"].mean(), base["B"]["plain"].mean()), flush=True)
res = []
for gus in (1.25, 1.5, 1.75, 2.0):
    for gin in (1.25, 1.5, 1.75, 2.0):
        if gus == gin == 1.5:
            continue
        a = run(gus, gin, S["A"])
        res.append((gus, gin, a["plain"].mean() - base["A"]["plain"].mean(), a["hard"].mean() - base["A"]["hard"].mean()))
        print(f"half A US {gus} India {gin}: plain {res[-1][2]:+.6f} hard {res[-1][3]:+.6f}", flush=True)
bg = max(res, key=lambda r: r[2] + r[3])
print("half-A pick", bg, flush=True)
bb = run(bg[0], bg[1], S["B"])
rng = np.random.default_rng(0)
for m in ("plain", "hard"):
    d = bb[m] - base["B"][m]; n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    print(f"half B {m}: diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(<=0) {np.mean(bs <= 0):.3f}", flush=True)
