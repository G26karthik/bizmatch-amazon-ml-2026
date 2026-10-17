"""Generator-operation residual model on the uncertain zone (local, no AWS).
Pairs: run28_7q best-S1 pairs with 0.02 < p < 0.98 (local_data/llm/unc_{val,test}_text.parquet: q, s1, prob, text_a = S1
"name | address", text_b = record). Features: logit(prob) + explicit raw-text operations between the two names and
addresses (legal-form relation, case pattern, accents, punctuation, brackets, token swap/add/drop with word kind,
typo similarity, web handle, initials, alias phrase, house/street/postcode relations, empty address, sources).
LightGBM fitted on validation half A (2-fold cross-fit by S1 for half-A scores); half B scored once through the
production decision (efopt gamma 1.5) on the reconstructed best-S1 table (see coll_val.py), paired bootstrap.
--test: refit on all validation pairs and write new test probabilities to local_data/llm/resid_test.parquet.

uv run --no-project --with-requirements requirements.txt python scripts/resid_model.py [--test] [--llm DIR] [--no_ops]
  --llm D1[,D2]: add each LoRA judge's logit (D/llm_val.parquet, D/llm_test.parquet: q, s1, logit) and logit_p x logit
  --no_ops : drop the raw-text operation features (logit_p [+ LLM] only)
  --extra NAME[,NAME2]: join label-free pair features from local_data/llm/NAME_val.parquet / NAME_test.parquet (q, s1, f...)
  --testlike : drop the records that exist only because of the 19% S1 hiding (copies of hidden S1s; test has none)
               from training, evaluation and the orphan set
  --dump_val FILE : write validation probabilities (q, s1, pn; half A cross-fit, half B from the half-A model)
  --ctx    : add context features: S1s / records sharing the record's and the S1's name key in the country, number of
             uncertain pairs competing for the same S1, their probability mass and this pair's rank among them
"""
import re
import sys
import unicodedata
from collections import Counter

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

SEED, VAL_MOD, HIDE = 42, 10, 19
CFG = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
CANON = {"ltd": "limited", "limited": "limited", "pvt": "private", "private": "private", "inc": "inc", "incorporated": "inc",
         "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "llc": "llc", "lp": "lp", "llp": "llp", "pc": "pc",
         "pllc": "pllc", "plc": "plc", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
         "snc": "snc", "ei": "ei"}
STOP = {"the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l", "&"}
GEN = {"center", "centre", "services", "service", "partners", "group", "solutions", "enterprises", "associates", "industries",
       "fils", "cie", "associes", "developpement", "groupe", "france", "frs", "freres", "compagnie", "sons", "brothers", "bros"}
ALIAS = re.compile(r"\b(dba|d/b/a|aka|a/k/a|t/a|f/k/a|trading as|doing business as)\b", re.I)
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()


def toks(s):
    return re.findall(r"[a-z0-9]+", asc(s).lower().replace(".", ""))


def casepat(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0
    if all(c.isupper() for c in letters):
        return 1
    if all(c.islower() for c in letters):
        return 2
    return 3


def feats(ta, tb, voc):
    na, _, aa = ta.partition(" | ")
    nb, _, ab = tb.partition(" | ")
    TA, TB = toks(na), toks(nb)
    la, lb = frozenset(CANON[t] for t in TA if t in CANON), frozenset(CANON[t] for t in TB if t in CANON)
    A, B = set(TA) - set(CANON) - STOP, set(TB) - set(CANON) - STOP
    lrel = 0 if la == lb else (1 if lb < la else (2 if la < lb else 3))
    add, drop = B - A, A - B
    kind_new = -1
    sim_swap = -1.0
    if len(add) == 1 and len(drop) == 1:
        w, o = next(iter(add)), next(iter(drop))
        kind_new = 0 if w in GEN else (1 if voc.get(w, 0) > 30 else 2)
        sim_swap = fuzz.ratio(w, o)
    elif len(add) == 1 and not drop:
        w = next(iter(add)); kind_new = 3 if w in GEN else (4 if voc.get(w, 0) > 30 else 5)
    raw_eq = na == nb
    ci_eq = na.lower() == nb.lower()
    ascii_eq = asc(na).lower() == asc(nb).lower()
    punct_eq = re.sub(r"[\W_]+", "", asc(na).lower()) == re.sub(r"[\W_]+", "", asc(nb).lower())
    order_eq = [t for t in TA if t not in CANON] == [t for t in TB if t not in CANON]
    handle = bool(re.match(r"^\s*[#@]?[\w&+\-]+(\.(com|fr|net|org|in|co))?\s*$", asc(nb).lower())) and (".com" in nb.lower() or nb.strip()[:1] in "#@")
    ini = "".join(t[0] for t in TA if t not in CANON and t not in STOP)
    initials = re.sub(r"[^a-z]", "", asc(nb).lower()) == ini and 1 < len(ini) <= 5
    brk_b, brk_a = nb.count("[") + nb.count("("), na.count("[") + na.count("(")
    alias = bool(ALIAS.search(nb))
    # address
    ea, eb = not aa.strip(), not ab.strip()
    TAa, TBa = toks(aa), toks(ab)
    ha = next((t for t in TAa if t.isdigit()), None); hb = next((t for t in TBa if t.isdigit()), None)
    dh = (int(hb) - int(ha)) if (ha and hb and len(ha) < 7 and len(hb) < 7) else None
    hrel = -1 if dh is None else (0 if dh == 0 else (1 if dh in OFFS else (2 if dh < 0 else 3)))
    sa, sb = set(TAa) - {ha}, set(TBa) - {hb}
    street_j = len(sa & sb) / max(1, len(sa | sb))
    pa = next((t for t in TAa if len(t) in (5, 6) and t.isdigit()), None); pb = next((t for t in TBa if len(t) in (5, 6) and t.isdigit()), None)
    post = -1 if not (pa and pb) else int(pa == pb)
    return [lrel, len(la), len(lb), kind_new, sim_swap, len(add), len(drop), int(raw_eq), int(ci_eq), int(ascii_eq), int(punct_eq), int(order_eq),
            casepat(na), casepat(nb), int(handle), int(initials), brk_a, brk_b, int(alias), int(ea), int(eb), hrel, street_j, post,
            fuzz.ratio(asc(na).lower(), asc(nb).lower()), fuzz.token_set_ratio(asc(na).lower(), asc(nb).lower()), len(na), len(nb)]


FN = ["lrel", "nla", "nlb", "kind_new", "sim_swap", "n_add", "n_drop", "raw_eq", "ci_eq", "ascii_eq", "punct_eq", "order_eq", "case_a",
      "case_b", "handle", "initials", "brk_a", "brk_b", "alias", "empty_a", "empty_b", "hrel", "street_j", "postal", "name_ratio",
      "name_tsr", "len_a", "len_b"]


LLMS = sys.argv[sys.argv.index("--llm") + 1].split(",") if "--llm" in sys.argv else []  # one or more judge dirs
LLM = bool(LLMS)
NO_OPS = "--no_ops" in sys.argv
CTX = "--ctx" in sys.argv
TESTLIKE = "--testlike" in sys.argv
NOISE_W = float(sys.argv[sys.argv.index("--noise_w") + 1]) if "--noise_w" in sys.argv else 1.0   # weight of unmatched records (test ~2x decoys)
EXTRA = sys.argv[sys.argv.index("--extra") + 1].split(",") if "--extra" in sys.argv else []
XN = []   # extra feature names, filled when the val table is joined
LEGK = set(CANON) | STOP


def nkey(s):
    return " ".join(sorted(set(toks(s)) - LEGK))


def ctx_feats(df, recdf):
    """recdf: rid, business_name, country, src for the split (train or test)."""
    keys = recdf.with_columns(pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k")).select("rid", "k", "country", "src")
    ns1 = keys.filter(pl.col("src") == 1).group_by("country", "k").len().rename({"len": "n_s1_key"})
    nrec = keys.filter(pl.col("src") != 1).group_by("country", "k").len().rename({"len": "n_rec_key"})
    kk = keys.join(ns1, on=["country", "k"], how="left").join(nrec, on=["country", "k"], how="left").fill_null(0)
    x = df.select("q", "s1", "prob").join(kk.select(pl.col("rid").alias("q"), pl.col("n_s1_key").alias("q_ns1"), pl.col("n_rec_key").alias("q_nrec")), on="q", how="left")           .join(kk.select(pl.col("rid").alias("s1"), pl.col("n_s1_key").alias("s_ns1"), pl.col("n_rec_key").alias("s_nrec")), on="s1", how="left")
    x = x.with_columns(pl.len().over("s1").alias("s1_ncand"), pl.col("prob").sum().over("s1").alias("s1_mass"),
                       pl.col("prob").rank("ordinal", descending=True).over("s1").alias("s1_rank"))
    return x.select("q_ns1", "q_nrec", "s_ns1", "s_nrec", "s1_ncand", "s1_mass", "s1_rank").fill_null(0).to_numpy().astype(np.float32)


def build(df, voc):
    p = np.clip(df["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    cols = [lp]
    for i in range(len(LLMS)):
        ll = df[f"logit{i}"].fill_null(float("nan")).to_numpy()   # pairs a judge did not score: missing, not 0
        cols += [ll, lp * ll]
    if not NO_OPS:
        cols.append(np.array([feats(a, b, voc) for a, b in zip(df["text_a"].to_list(), df["text_b"].to_list())], dtype=np.float32))
    if CTX:
        cols.append(ctx_feats(df, CTX_REC))
    if XN:
        cols.append(df.select(XN).to_numpy().astype(np.float32))
    return np.column_stack(cols)


def join_extra(df, part):
    for nm_ in EXTRA:
        e = pl.read_parquet(f"local_data/llm/{nm_}_{part}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
        cols_ = [c for c in e.columns if c not in ("q", "s1")]
        e = e.rename({c: f"{nm_}__{c}" for c in cols_})
        df = df.join(e, on=["q", "s1"], how="left")
        if part == "val":
            XN.extend(f"{nm_}__{c}" for c in cols_)
    return df


# ---- labels and halves (as coll_val.py)
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]) \
        .with_row_index("rid")
rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
_tp = pl.read_parquet("local_data/train_pairs.parquet")
_hid = rec.filter(pl.col("hidden")).select(pl.col("entity_id").alias("s1"))
HSET = set(rec.join(_tp.join(_hid, on="s1").select(pl.col("m").alias("entity_id")), on="entity_id")["rid"].to_list()) if TESTLIKE else set()
gt = _tp.join(_hid, on="s1", how="anti")
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
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list()) - HSET
valS1 = set(S["A"]) | set(S["B"])
half = dict(lab.filter(pl.col("src") == 1).select("rid", "half_a").iter_rows())
voc = Counter(w for n in rec.filter(pl.col("src") == 1)["business_name"].to_list() for w in set(toks(n)))
CTX_REC = live.select(pl.col("rid").cast(pl.UInt32), "business_name", "country", "src") if CTX else None
del rec, live, grp
print("labels ready; half B S1s", len(S["B"]), flush=True)

uvt = pl.read_parquet("local_data/llm/unc_val_text.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
for i, d in enumerate(LLMS):
    lv = pl.read_parquet(f"{d}/llm_val.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("logit").alias(f"logit{i}"))
    uvt = uvt.join(lv, on=["q", "s1"], how="left")
    print(f"judge {d} logits joined for val:", uvt[f"logit{i}"].is_not_null().sum(), "of", uvt.height, flush=True)
uvt = join_extra(uvt, "val")
if TESTLIKE:
    uvt = uvt.filter(~pl.col("q").is_in(list(HSET)))
    print("test-like: H records removed, uncertain val pairs left", uvt.height, flush=True)
print("extra features:", XN, flush=True)
FNAMES = ["logit_p"] + [f for i in range(len(LLMS)) for f in (f"llm{i}", f"lp_x_llm{i}")] + ([] if NO_OPS else FN) +     (["q_ns1", "q_nrec", "s_ns1", "s_nrec", "s1_ncand", "s1_mass", "s1_rank"] if CTX else []) + XN
uv = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
if TESTLIKE:
    uv = uv.filter(~pl.col("q").is_in(list(HSET)))
uvt = uvt.with_columns(pl.Series("y", [par.get(q) == s for q, s in zip(uvt["q"].to_list(), uvt["s1"].to_list())]).cast(pl.Int8),
                       pl.Series("hA", [half.get(s, False) for s in uvt["s1"].to_list()]))
X = build(uvt, voc); y = uvt["y"].to_numpy(); hA = uvt["hA"].to_numpy()
W = np.where(np.isin(uvt["q"].to_numpy(), np.fromiter(orph, dtype=np.int64)), NOISE_W, 1.0) if NOISE_W != 1.0 else np.ones(len(y))
print("noise weight", NOISE_W, "rows weighted", int((W != 1.0).sum()), flush=True)
print("features", X.shape, "positives", y.mean().round(3), flush=True)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=6)
s1v = uvt["s1"].to_numpy()
fold = (s1v % 2).astype(int)
newp = np.full(len(y), np.nan)
iA = np.flatnonzero(hA)
for f in (0, 1):  # cross-fit inside half A
    tr, te = iA[fold[iA] != f], iA[fold[iA] == f]
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=W[tr], feature_name=FNAMES), num_boost_round=400)
    newp[te] = m.predict(X[te])
mA = lgb.train(params, lgb.Dataset(X[iA], y[iA], weight=W[iA], feature_name=FNAMES), num_boost_round=400)
iB = np.flatnonzero(~hA)
newp[iB] = mA.predict(X[iB])
ll = lambda yy, pp: -np.mean(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1)))  # noqa: E731
p0 = uvt["prob"].to_numpy()
print(f"logloss half A (OOF) {ll(y[iA], p0[iA]):.4f} -> {ll(y[iA], newp[iA]):.4f};  half B {ll(y[iB], p0[iB]):.4f} -> {ll(y[iB], newp[iB]):.4f}", flush=True)
imp = sorted(zip(FNAMES, mA.feature_importance("gain")), key=lambda t: -t[1])[:12]
print("top gain:", [(n, round(g)) for n, g in imp], flush=True)

uq = set(uv["q"].to_list())
cert = [(q, p, 0.999) for q, p in par.items() if p in valS1 and q not in uq]
certdf = pl.DataFrame(cert, schema={"q": pl.UInt32, "s1": pl.UInt32, "prob": pl.Float64}, orient="row")
base_best = pl.concat([uv.select("q", "s1", pl.col("prob").cast(pl.Float64)), certdf])
newdf = uvt.select("q", "s1").with_columns(pl.Series("pn", newp))
if "--dump_val" in sys.argv:
    newdf.write_parquet(sys.argv[sys.argv.index("--dump_val") + 1]); print("dumped val probabilities", newdf.height, flush=True)


def best_with(alpha):  # blend of old and new probabilities (alpha = weight of the new model)
    m = uv.join(newdf, on=["q", "s1"], how="left").with_columns(
        pl.when(pl.col("pn").is_not_null()).then(alpha * pl.col("pn") + (1 - alpha) * pl.col("prob")).otherwise(pl.col("prob")).cast(pl.Float64).alias("prob"))
    return pl.concat([m.select("q", "s1", "prob"), certdf])


def per_entity(bst, s_list):
    ch = _chosen(bst, CFG)
    plain, hard = {}, {}
    for q, s in ch.select("q", "s1").iter_rows():
        plain.setdefault(s, set()).add(q)
        hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s_list]) for m, pm in (("plain", plain), ("hard", hard))}


base = {h: per_entity(base_best, S[h]) for h in "AB"}
res = []
for alpha in (0.25, 0.5, 0.75, 1.0):
    a = per_entity(best_with(alpha), S["A"])
    res.append((alpha, a["plain"].mean() - base["A"]["plain"].mean(), a["hard"].mean() - base["A"]["hard"].mean()))
    print(f"half A alpha {alpha}: plain {res[-1][1]:+.6f} hard {res[-1][2]:+.6f}", flush=True)
alpha = max(res, key=lambda r: r[1] + r[2])[0]
print("half-A pick alpha", alpha, flush=True)
bb = per_entity(best_with(alpha), S["B"])
rng = np.random.default_rng(0)
for mname in ("plain", "hard"):
    d = bb[mname] - base["B"][mname]; n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    print(f"half B {mname}: {base['B'][mname].mean():.6f} -> {bb[mname].mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(<=0) {np.mean(bs <= 0):.3f}", flush=True)
if "--test" in sys.argv:
    mall = lgb.train(params, lgb.Dataset(X, y, weight=W, feature_name=FNAMES), num_boost_round=400)
    trec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)])              .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    tvoc = Counter(w for n in trec["business_name"].head(1732544).to_list() for w in set(toks(n)))  # S1 rows come first
    CTX_REC = trec.select("rid", "business_name", "country", "src") if CTX else None
    del trec
    ut = pl.read_parquet("local_data/llm/unc_test_text.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    for i, d in enumerate(LLMS):
        lt = pl.read_parquet(f"{d}/llm_test.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("logit").alias(f"logit{i}"))
        ut = ut.join(lt, on=["q", "s1"], how="left")
    ut = join_extra(ut, "test")
    pt = mall.predict(build(ut, tvoc))
    pn = alpha * pt + (1 - alpha) * ut["prob"].to_numpy()
    ut.select("q", "s1", "prob").with_columns(pl.Series("pn", pn), pl.lit(alpha).alias("alpha")).write_parquet("local_data/llm/resid_test.parquet")
    print("wrote local_data/llm/resid_test.parquet", ut.height, flush=True)
