"""End-to-end entity resolution pipeline (reverse assignment: each S2/S3 record -> best S1 parent or none).

Stages (each writes Parquet under --work):
  prep      normalize text fields               -> {split}/records.parquet
  retrieve  per-country union of 3 TF-IDF retrievers -> {split}/pairs_{country}.parquet
  features  pairwise similarity features         -> {split}/feats/*.parquet
  train     LightGBM on train split (val S1 groups held out), tune threshold on val macro F0.5
  predict   score test, assign, write output/matching_results.tsv + candidate_pairs.tsv

python -m ber.pipeline all --data /work/data --work /work/run --out /work/output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from ber.metric import entity_f05
from ber.normalize import addr_tokens, name_core, name_skel

NPROC = os.cpu_count() or 4
K_COMB, K_ADDR, K_NAME = 10, 15, 8   # union of three retrievers per query
K_NAME_EMPTY = 40                    # name-only depth for queries with an empty address (v2 recall there: 54%)
DF_CAP = 3000          # drop S1-side features shared by more S1 records than this (speed; weak evidence)
W_ADDR, W_NAME = 0.6, 0.4
N_FEAT = 2 ** 23
VAL_MOD = 10           # S1 groups with hash % VAL_MOD == 0 are validation
SEED = 42
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- prep
def _norm_chunk(rows):
    out = []
    for name, addr in rows:
        at = addr_tokens(addr)
        nums = [t for t in at if t[:1].isdigit()]
        words = [t for t in at if not t[:1].isdigit()]
        postal = next((t for t in reversed(nums) if t.isdigit() and len(t) in (5, 6)), "")
        out.append((name_core(name), " ".join(at), nums[0] if nums else "", postal, nums,
                    any(ord(c) > 127 and c.isalpha() for c in name), name_skel(name), " ".join(words)))
    return out


def prep(data: Path, work: Path, split: str) -> None:
    dfs = []
    for s in (1, 2, 3):
        df = pl.read_parquet(data / f"{split}_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src"))
        dfs.append(df)
    df = pl.concat(dfs)
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    step = math.ceil(len(rows) / (NPROC * 8))
    with Pool(NPROC) as p:
        res = [r for part in p.map(_norm_chunk, [rows[i:i + step] for i in range(0, len(rows), step)]) for r in part]
    cols = list(zip(*res))
    del res
    df = df.with_columns(
        pl.Series("name_n", cols[0]), pl.Series("addr_n", cols[1]), pl.Series("house", cols[2]),
        pl.Series("postal", cols[3]), pl.Series("nums", cols[4], dtype=pl.List(pl.String)),
        pl.Series("name_nonascii", cols[5]), pl.Series("name_sk", cols[6]), pl.Series("addr_w", cols[7]),
    ).with_columns((pl.col("business_address").str.strip_chars() == "").alias("addr_empty"))
    (work / split).mkdir(parents=True, exist_ok=True)
    df.write_parquet(work / split / "records.parquet")
    log(f"prep {split}: {df.height} records")


# ---------------------------------------------------------------- retrieve
def _docs(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    addr_docs, name_docs = [], []
    for ad, nm, sk in zip(df["addr_n"].to_list(), df["name_n"].to_list(), df["name_sk"].to_list()):
        t = ad.split()
        addr_docs.append(" ".join(t + [f"{x}_{y}" for x, y in zip(t, t[1:])]))
        s = "_" + nm.replace(" ", "_") + "_"
        name_docs.append(" ".join(nm.split() + ["#" + s[i:i + 3] for i in range(len(s) - 2)]
                                  + ["~" + k for k in sk.split()]))
    return addr_docs, name_docs


def _tfidf(docs1, docs2):
    """Returns (X1 normalized, X2 normalized, X1 with high-df S1 columns removed for retrieval)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    import scipy.sparse as sp
    hv = HashingVectorizer(n_features=N_FEAT, token_pattern=r"\S+", lowercase=False, norm=None,
                           alternate_sign=False, binary=True, dtype=np.float32)
    X1, X2 = hv.transform(docs1), hv.transform(docs2)
    df1 = np.bincount(X1.indices, minlength=N_FEAT)
    df = df1 + np.bincount(X2.indices, minlength=N_FEAT)
    idf = (np.log((X1.shape[0] + X2.shape[0] + 1) / (df + 1)) + 1).astype(np.float32)
    D = sp.diags(idf)
    X1, X2 = normalize(X1 @ D).tocsr(), normalize(X2 @ D).tocsr()
    X1c = (X1 @ sp.diags((df1 <= DF_CAP).astype(np.float32))).tocsr()
    X1c.eliminate_zeros()
    return X1, X2, X1c


def _topk(Q, S1T, k, name):
    from sparse_dot_topn import sp_matmul_topn
    C = sp_matmul_topn(Q, S1T, top_n=k, threshold=0.01, sort=True, n_threads=NPROC).tocsr()
    counts = np.diff(C.indptr)
    qi = np.repeat(np.arange(C.shape[0]), counts)
    rank = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([], int)
    return pl.DataFrame({"ql": qi.astype(np.int32), "sl": C.indices.astype(np.int32), name: rank.astype(np.int8)})


def _rowdot(X2, X1, qi, si):
    out = np.empty(len(qi), np.float32)
    step = 2_000_000
    for j in range(0, len(qi), step):
        out[j:j + step] = np.asarray(X2[qi[j:j + step]].multiply(X1[si[j:j + step]]).sum(1)).ravel()
    return out


def retrieve(work: Path, split: str) -> None:
    import scipy.sparse as sp
    rec = pl.read_parquet(work / split / "records.parquet",
                          columns=["entity_id", "src", "country", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
    for old in (work / split).glob("pairs_*.parquet"):
        old.unlink()
    pct = hide_pct(work) if split == "train" else 0
    for country in rec["country"].unique().sort().to_list():
        r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))
        if pct:
            r1 = r1.filter(~hidden_s1(pl.col("entity_id"), pct))
        r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))
        if r1.height == 0 or r2.height == 0:
            continue
        a1, n1 = _docs(r1)
        a2, n2 = _docs(r2)
        A1, A2, A1c = _tfidf(a1, a2)
        N1, N2, N1c = _tfidf(n1, n2)
        del a1, a2, n1, n2
        wa, wn = math.sqrt(W_ADDR), math.sqrt(W_NAME)
        CT = sp.hstack([A1c * wa, N1c * wn]).T.tocsr()
        Q = sp.hstack([A2 * wa, N2 * wn]).tocsr()
        AT, NT = A1c.T.tocsr(), N1c.T.tocsr()
        del A1c, N1c
        log(f"retrieve {split}/{country}: S1={r1.height} Q={r2.height}")
        outs = []
        step = 200_000
        for i in range(0, Q.shape[0], step):
            c = _topk(Q[i:i + step], CT, K_COMB, "rank_c")
            a = _topk(A2[i:i + step], AT, K_ADDR, "rank_a")
            n = _topk(N2[i:i + step], NT, K_NAME, "rank_n")
            empty = np.flatnonzero(np.diff(A2[i:i + step].indptr) == 0)
            if len(empty):
                e = _topk(N2[i:i + step][empty], NT, K_NAME_EMPTY, "rank_n")
                e = e.with_columns(pl.Series("ql", empty[e["ql"].to_numpy()].astype(np.int32)))
                n = pl.concat([n, e]).group_by(["ql", "sl"]).agg(pl.col("rank_n").min())
            u = (c.join(a, on=["ql", "sl"], how="full", coalesce=True)
                  .join(n, on=["ql", "sl"], how="full", coalesce=True)
                  .with_columns(pl.col("ql") + i, *[pl.col(x).fill_null(99) for x in ("rank_c", "rank_a", "rank_n")]))
            outs.append(u)
        u = pl.concat(outs)
        del outs, Q, CT, AT, NT
        qi, si = u["ql"].to_numpy(), u["sl"].to_numpy()
        cos_a, cos_n = _rowdot(A2, A1, qi, si), _rowdot(N2, N1, qi, si)
        rid1, rid2 = r1["rid"].to_numpy(), r2["rid"].to_numpy()
        pairs = pl.DataFrame({"q": rid2[qi], "s1": rid1[si], "cos_a": cos_a, "cos_n": cos_n}).with_columns(
            (W_ADDR * pl.col("cos_a") + W_NAME * pl.col("cos_n")).alias("score"),
            u["rank_c"], u["rank_a"], u["rank_n"])
        pairs.write_parquet(work / split / f"pairs_{country}.parquet")
        log(f"retrieve {split}/{country}: {pairs.height} pairs ({pairs.height / r2.height:.1f}/query)")


def add_neural(work: Path, split: str) -> None:
    """Union neural-encoder candidates ({split}/neural_pairs.parquet: q, s1, emb_cos, rank_e; from ber.neural on GPU)
    into pairs_{country}.parquet, computing TF-IDF cosines for the new pairs. emb_cos for *all* pairs is joined later in
    features from {split}/emb_cos.parquet."""
    import scipy.sparse as sp  # noqa: F401
    rec = pl.read_parquet(work / split / "records.parquet",
                          columns=["entity_id", "src", "country", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
    neu = pl.read_parquet(work / split / "neural_pairs.parquet").with_columns(pl.col("q").cast(pl.UInt32),
                                                                                pl.col("s1").cast(pl.UInt32))
    for pf in sorted((work / split).glob("pairs_*.parquet")):
        country = pf.stem.removeprefix("pairs_")
        p = pl.read_parquet(pf)
        if "rank_e" in p.columns:
            log(f"add_neural {split}/{country}: already merged")
            continue
        r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))
        if split == "train" and hide_pct(work):
            r1 = r1.filter(~hidden_s1(pl.col("entity_id"), hide_pct(work)))
        r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))
        rid1, rid2 = r1["rid"].to_numpy(), r2["rid"].to_numpy()
        n = neu.filter(pl.col("q").is_in(pl.Series(rid2).implode()) & pl.col("s1").is_in(pl.Series(rid1).implode()))
        new = n.join(p.select("q", "s1"), on=["q", "s1"], how="anti")
        if new.height:
            a1, n1 = _docs(r1)
            a2, n2 = _docs(r2)
            A1, A2, _ = _tfidf(a1, a2)
            N1, N2, _ = _tfidf(n1, n2)
            pos1 = np.full(int(rec.height), -1, np.int64)
            pos2 = np.full(int(rec.height), -1, np.int64)
            pos1[rid1], pos2[rid2] = np.arange(len(rid1)), np.arange(len(rid2))
            qi, si = pos2[new["q"].to_numpy()], pos1[new["s1"].to_numpy()]
            new = new.select("q", "s1").with_columns(pl.Series("cos_a", _rowdot(A2, A1, qi, si)),
                                                     pl.Series("cos_n", _rowdot(N2, N1, qi, si)))
            new = new.with_columns((W_ADDR * pl.col("cos_a") + W_NAME * pl.col("cos_n")).alias("score"),
                                   *[pl.lit(99, dtype=pl.Int8).alias(c) for c in ("rank_c", "rank_a", "rank_n")])
        u = pl.concat([p.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)),
                       new.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
                          .select(p.columns)], how="vertical_relaxed")
        u = u.join(n.select("q", "s1", "rank_e"), on=["q", "s1"], how="left").with_columns(pl.col("rank_e").fill_null(99))
        u.write_parquet(pf)
        log(f"add_neural {split}/{country}: +{new.height} neural-only pairs -> {u.height} ({u.height / r2.height:.1f}/query)")


# ---------------------------------------------------------------- features
FEATS = ["score", "cos_a", "cos_n", "rank", "rank_c", "rank_a", "rank_n", "top1", "rel_top1", "gap12",
         "top1_a", "top1_n", "d_top1_a", "d_top1_n", "n_cand",
         "s1_in_rank", "s1_in_n", "s1_best_other", "s1_n_r0",
         "name_ratio", "name_tsr", "name_pr", "name_jw", "skel_ratio", "skel_tsr", "first_tok_eq",
         "addr_ratio", "addr_tsr", "addr_pr", "street_tsr", "street_ratio",
         "house_eq", "house_miss", "house_sim", "house_absdiff", "postal_eq", "postal_miss", "num_jac", "num_inter",
         "q_nonascii", "q_addr_empty", "q_src3", "len_name1", "len_name2", "len_addr1", "len_addr2"]
NEURAL_FEATS = ["emb_cos", "rank_e", "emb_top1", "d_emb_top1", "emb_rank"]


STR_COLS = ["name_ratio", "name_tsr", "name_pr", "name_jw", "skel_ratio", "skel_tsr", "addr_ratio", "addr_tsr", "addr_pr",
            "street_tsr", "street_ratio", "house_sim", "house_eq", "house_miss", "house_absdiff", "postal_eq",
            "postal_miss", "num_jac", "num_inter", "first_tok_eq", "q_nonascii", "q_addr_empty", "q_src3",
            "len_name1", "len_name2", "len_addr1", "len_addr2"]


def _records_for_features(work: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(work / split / "records.parquet",
                           columns=["name_n", "addr_n", "house", "postal", "nums", "name_nonascii", "addr_empty", "src",
                                    "name_sk", "addr_w"]).with_row_index("rid")


def features(work: Path, split: str) -> None:
    """Full feature build: every pair gets string features + context. Parts are aligned to query boundaries."""
    rec = _records_for_features(work, split)
    fd = work / split / "feats"
    fd.mkdir(exist_ok=True)
    resume = os.environ.get("FEATURES_RESUME") == "1"  # keep finished parts (spot interruptions); partition is deterministic
    if not resume:
        for old in fd.glob("*.parquet"):
            old.unlink()
    for pf in sorted((work / split).glob("pairs_*.parquet")):
        country = pf.stem.removeprefix("pairs_")
        p = _context(work, split, pl.read_parquet(pf))
        p = p.sort(["q", "s1"]).with_columns((pl.int_range(pl.len()) // 5_000_000).alias("_c"))
        p = p.with_columns(pl.col("_c").first().over("q"))
        for ci, c in enumerate(p.partition_by("_c", include_key=False, maintain_order=True)):
            out = fd / f"{country}_part{ci:03d}.parquet"
            if resume and out.exists():
                continue
            feat = c.join(_pair_strings(c, rec), on=["q", "s1"], how="left")
            feat.write_parquet(out.with_suffix(".tmp"))
            out.with_suffix(".tmp").rename(out)
            log(f"features {split}/{country}: part {ci} rows {feat.height}")


def features_incremental(work: Path, split: str) -> None:
    """After add_neural: reuse string features already in the feats parts, compute them only for new pairs, and
    rebuild every part with fresh pair/context columns. Each query keeps all its candidates in one part."""
    rec = _records_for_features(work, split)
    fd = work / split / "feats"
    for pf in sorted((work / split).glob("pairs_*.parquet")):
        country = pf.stem.removeprefix("pairs_")
        parts = sorted(fd.glob(f"{country}_part*.parquet"))
        p = _context(work, split, pl.read_parquet(pf))
        old_q = pl.concat([pl.read_parquet(x, columns=["q"]).unique().with_columns(pl.lit(i).alias("_c"),
                                                                                    pl.col("q").cast(pl.UInt32))
                           for i, x in enumerate(parts)])
        p = p.join(old_q, on="q", how="left").with_columns(pl.col("_c").fill_null(len(parts) - 1))
        n_new = 0
        for i, x in enumerate(parts):
            ctx = p.filter(pl.col("_c") == i).drop("_c")
            old = pl.read_parquet(x, columns=["q", "s1"] + STR_COLS).with_columns(pl.col("q").cast(pl.UInt32),
                                                                                  pl.col("s1").cast(pl.UInt32))
            new_pairs = ctx.select("q", "s1").join(old.select("q", "s1"), on=["q", "s1"], how="anti")
            strings = pl.concat([old, _pair_strings(new_pairs, rec)], how="vertical_relaxed") if new_pairs.height else old
            ctx.join(strings, on=["q", "s1"], how="left").write_parquet(x)
            n_new += new_pairs.height
            log(f"features+ {split}/{country}: part {i} rows {ctx.height} (+{new_pairs.height} new)")
        log(f"features+ {split}/{country}: {n_new} new pairs")


def _context(work: Path, split: str, p: pl.DataFrame) -> pl.DataFrame:
    """Cheap per-query / per-S1 context columns over the full candidate set of one country."""
    p = p.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    ec = work / split / "emb_cos.parquet"
    if ec.exists():  # neural similarity for every pair (computed on GPU by ber.neural score)
        e = pl.read_parquet(ec).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
        p = p.join(e, on=["q", "s1"], how="left")
        p = p.with_columns(pl.col("emb_cos").fill_null(-1.0),
                           pl.col("emb_cos").max().over("q").alias("emb_top1"),
                           (pl.col("emb_cos").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16).alias("emb_rank"))
        p = p.with_columns((pl.col("emb_top1") - pl.col("emb_cos")).alias("d_emb_top1"))
    if True:
        p = p.with_columns(
            (pl.col("score").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16).alias("rank"),
            pl.col("score").max().over("q").alias("top1"),
            pl.col("cos_a").max().over("q").alias("top1_a"),
            pl.col("cos_n").max().over("q").alias("top1_n"),
            pl.len().over("q").alias("n_cand"),
        )
        second = p.filter(pl.col("rank") == 1).select("q", pl.col("score").alias("top2"))
        p = p.join(second, on="q", how="left").with_columns(
            (pl.col("score") / pl.col("top1")).alias("rel_top1"),
            (pl.col("top1") - pl.col("top2").fill_null(0)).alias("gap12"),
            (pl.col("top1_a") - pl.col("cos_a")).alias("d_top1_a"),
            (pl.col("top1_n") - pl.col("cos_n")).alias("d_top1_n"),
            pl.col("score").rank("ordinal", descending=True).over("s1").alias("s1_in_rank"),
            pl.len().over("s1").alias("s1_in_n"),
            pl.when(pl.col("rank") == 0).then(pl.col("score")).otherwise(None).alias("_r0"),
        ).drop("top2")
        p = p.with_columns(pl.col("_r0").max().over("s1").fill_null(0).alias("s1_best_other"),
                           pl.col("_r0").is_not_null().sum().over("s1").alias("s1_n_r0")).drop("_r0")
    return p


def _pair_strings(c: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Expensive string-similarity features for pairs c; returns q, s1 + STR_COLS."""
    from rapidfuzz import fuzz, distance
    from rapidfuzz.process import cpdist
    c = c.select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    if True:
        if True:
            a = c.select("s1").join(rec.with_columns(pl.col("rid").cast(pl.UInt32)), left_on="s1", right_on="rid", how="left")
            b = c.select("q").join(rec.with_columns(pl.col("rid").cast(pl.UInt32)), left_on="q", right_on="rid", how="left")
            n1, n2 = a["name_n"].to_list(), b["name_n"].to_list()
            k1, k2 = a["name_sk"].to_list(), b["name_sk"].to_list()
            d1, d2 = a["addr_n"].to_list(), b["addr_n"].to_list()
            w1, w2 = a["addr_w"].to_list(), b["addr_w"].to_list()
            h1s, h2s = a["house"].to_list(), b["house"].to_list()
            f = {
                "name_ratio": cpdist(n1, n2, scorer=fuzz.ratio, workers=-1),
                "name_tsr": cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1),
                "name_pr": cpdist(n1, n2, scorer=fuzz.partial_ratio, workers=-1),
                "name_jw": cpdist(n1, n2, scorer=distance.JaroWinkler.normalized_similarity, workers=-1),
                "skel_ratio": cpdist(k1, k2, scorer=fuzz.ratio, workers=-1),
                "skel_tsr": cpdist(k1, k2, scorer=fuzz.token_set_ratio, workers=-1),
                "addr_ratio": cpdist(d1, d2, scorer=fuzz.ratio, workers=-1),
                "addr_tsr": cpdist(d1, d2, scorer=fuzz.token_set_ratio, workers=-1),
                "addr_pr": cpdist(d1, d2, scorer=fuzz.partial_ratio, workers=-1),
                "street_tsr": cpdist(w1, w2, scorer=fuzz.token_set_ratio, workers=-1),
                "street_ratio": cpdist(w1, w2, scorer=fuzz.ratio, workers=-1),
                "house_sim": cpdist(h1s, h2s, scorer=distance.Levenshtein.normalized_similarity, workers=-1),
            }
            h1, h2 = a["house"], b["house"]
            z1, z2 = a["postal"], b["postal"]
            miss = (h1 == "") | (h2 == "")
            hn1 = h1.str.extract(r"^(\d+)$").cast(pl.Float64, strict=False)
            hn2 = h2.str.extract(r"^(\d+)$").cast(pl.Float64, strict=False)
            absdiff = ((hn1 - hn2).abs() + 1).log().fill_null(-1)
            inter = a["nums"].list.set_intersection(b["nums"]).list.len()
            union = a["nums"].list.set_union(b["nums"]).list.len()
            f["house_sim"] = np.where(miss.to_numpy(), -1, f["house_sim"])
            feat = c.with_columns(
                *[pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in f.items()],
                ((h1 == h2) & (h1 != "")).alias("house_eq"),
                miss.alias("house_miss"),
                absdiff.cast(pl.Float32).alias("house_absdiff"),
                ((z1 == z2) & (z1 != "")).alias("postal_eq"),
                ((z1 == "") | (z2 == "")).alias("postal_miss"),
                (inter / union.clip(lower_bound=1)).cast(pl.Float32).alias("num_jac"),
                inter.alias("num_inter"),
                (a["name_n"].str.split(" ").list.first() == b["name_n"].str.split(" ").list.first()).alias("first_tok_eq"),
                b["name_nonascii"].alias("q_nonascii"),
                b["addr_empty"].alias("q_addr_empty"),
                (b["src"] == 3).alias("q_src3"),
                a["name_n"].str.len_chars().alias("len_name1"),
                b["name_n"].str.len_chars().alias("len_name2"),
                a["addr_n"].str.len_chars().alias("len_addr1"),
                b["addr_n"].str.len_chars().alias("len_addr2"),
            )
    return feat


# ---------------------------------------------------------------- labels / split
def hide_pct(work: Path) -> int:
    """Percent of *training* S1 entities hidden from the S1 pool (work/config.json {"hide_pct": N}).

    Test has ~5.8 S2/S3 records per S1 vs 4.7 in train, i.e. ~40% unmatched records vs 26%, consistent with test
    containing records of businesses absent from S1. Hiding 19% of training S1 entities turns their S2/S3 records into
    such orphans and reproduces the test density (4.68 / 0.81 = 5.8), so training and validation look like test."""
    cfg = work / "config.json"
    return int(json.loads(cfg.read_text()).get("hide_pct", 0)) if cfg.exists() else 0


def hidden_s1(entity_id: pl.Expr, pct: int) -> pl.Expr:
    return (entity_id.hash(SEED + 77) % 100) < pct


def labels(data: Path, work: Path) -> pl.DataFrame:
    """rid-level: for each train record, its true parent S1 rid (or null) and its split group.
    Hidden S1 entities are dropped and their records become unmatched (see hide_pct)."""
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
    gt = pl.read_parquet(data / "train_pairs.parquet")
    pct = hide_pct(work)
    if pct:
        gt = gt.filter(~hidden_s1(pl.col("s1"), pct))
        rec = rec.filter(~((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), pct)))
    id2rid = rec.select("entity_id", "rid")
    parent = (gt.join(id2rid.rename({"entity_id": "m", "rid": "q"}), on="m")
                .join(id2rid.rename({"entity_id": "s1", "rid": "parent"}), on="s1").select("q", "parent"))
    grp = rec.with_columns((pl.col("entity_id").hash(SEED) % VAL_MOD == 0).alias("own_val"))
    lab = grp.join(parent, left_on="rid", right_on="q", how="left")
    lab = lab.join(grp.select(pl.col("rid").alias("parent"), pl.col("own_val").alias("parent_val")), on="parent", how="left")
    return lab.with_columns(pl.coalesce("parent_val", "own_val").alias("is_val")).select("rid", "src", "entity_id", "parent", "is_val")


def _parts(work: Path, split: str) -> list[Path]:
    return sorted((work / split / "feats").glob("*.parquet"))


def feat_names(df: pl.DataFrame) -> list[str]:
    return FEATS + [c for c in NEURAL_FEATS if c in df.columns]


def _to_x(df: pl.DataFrame, names: list[str] | None = None) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in (names or feat_names(df))]).to_numpy()


def _best_per_query(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("prob", descending=True).group_by("q", maintain_order=True).first().select("q", "s1", "prob")


def decide_pairs(best: pl.DataFrame, miss: float, floor: float = 0.02) -> pl.DataFrame:
    """Expected-F0.5 subset per S1 over the queries whose argmax is that S1."""
    from ber.decide import select
    b = best.filter(pl.col("prob") >= floor).sort(["s1", "prob"], descending=[False, True])
    if b.height == 0:
        return b.select("q", "s1")
    starts = np.concatenate([[0], np.flatnonzero(np.diff(b["s1"].to_numpy())) + 1, [b.height]]).astype(np.int64)
    keep = select(starts, b["prob"].to_numpy().astype(np.float64), float(miss))
    return b.filter(pl.Series(keep)).select("q", "s1")


def macro_on(s1_rids: list, truth: dict, pred) -> float:
    pm: dict = {}
    for q, s1 in pred:
        pm.setdefault(s1, set()).add(q)
    return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in s1_rids]))


PRUNE_P = 0.003  # stage-A probability below which a candidate pair is discarded before the main model
PARAMS_A = dict(objective="binary", learning_rate=0.15, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
                bagging_fraction=0.7, bagging_freq=1, num_threads=NPROC, verbose=-1, seed=SEED)
PARAMS_B = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                bagging_fraction=0.8, bagging_freq=1, num_threads=NPROC, verbose=-1, seed=SEED)


def _cascade(f: pl.DataFrame, model_a, model_b) -> pl.DataFrame:
    """Stage A (small, fast) prunes weak candidates; stage B scores the survivors. Returns survivors with prob."""
    pa = model_a.predict(_to_x(f), num_threads=NPROC).astype(np.float32)
    f = f.with_columns(pl.Series("pA", pa)).filter(pl.col("pA") >= PRUNE_P)
    if f.height == 0:
        return f.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("prob"))
    return f.with_columns(pl.Series("prob", model_b.predict(_to_x(f), num_threads=NPROC).astype(np.float32)))


def train_a(data: Path, work: Path) -> None:
    """Only the stage-A pruning model (used by ber.stage2.prune); skips the stage-B single-model evaluation."""
    import lightgbm as lgb
    lab = labels(data, work)
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val")
    samp = pl.col("q").hash(SEED + 1) % 6 == 0
    tr = pl.concat([pl.read_parquet(pp).join(qlab, on="q", how="left").filter(samp & ~pl.col("is_val"))
                    for pp in _parts(work, "train")])
    tr = tr.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))
    log(f"stage A train rows {tr.height} pos {tr['y'].sum()}")
    model_a = lgb.train(PARAMS_A, lgb.Dataset(_to_x(tr), tr["y"].to_numpy().astype(np.int8), feature_name=feat_names(tr)),
                        num_boost_round=150)
    model_a.save_model(str(work / "modelA.txt"))
    log("stage A saved")


def _eval_block(name, s1_list, truth, best, rep):
    """Threshold sweep + expected-F0.5 selection on one S1 set; returns (best_f05, thr) of the sweep."""
    s1s = pl.Series(s1_list)
    b = best.filter(pl.col("s1").is_in(s1s.implode()))
    top = (0.0, 0.5)
    for thr in np.arange(0.3, 0.96, 0.05):
        sc = macro_on(s1_list, truth, b.filter(pl.col("prob") >= thr).select("q", "s1").iter_rows())
        rep[f"{name}_thr{thr:.2f}"] = sc
        top = max(top, (sc, round(float(thr), 2)))
    ef = macro_on(s1_list, truth, decide_pairs(b, 0.0).iter_rows())
    rep[f"{name}_best_thr"], rep[f"{name}_f05_thr"], rep[f"{name}_f05_efopt"] = top[1], top[0], ef
    log(f"{name}: n={len(s1_list)} thr={top[1]} f05={top[0]:.5f} efopt={ef:.5f}")
    return top, ef


def _score_at(s1_list, truth, best, cfg):
    s1s = pl.Series(s1_list)
    b = best.filter(pl.col("s1").is_in(s1s.implode()))
    chosen = decide_pairs(b, cfg.get("miss", 0.0)) if cfg["mode"] == "efopt" else         b.filter(pl.col("prob") >= cfg["thr"]).select("q", "s1")
    return macro_on(s1_list, truth, chosen.iter_rows())


def train(data: Path, work: Path, max_rounds: int = 2000) -> None:
    """Train with early stopping on val-A pairs; choose decision rule on val-A; report untouched val-B.

    Overfitting checks: val-B never influences any choice; train-S1 sample F0.5 vs val-B F0.5 shows the gap.
    """
    import lightgbm as lgb
    lab = labels(data, work)
    lab = lab.with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    qlab = lab.select(pl.col("rid").alias("q"), "parent", "is_val", "half_a")

    def labelled(path):
        f = pl.read_parquet(path).join(qlab, on="q", how="left")
        return f.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"))

    parts = _parts(work, "train")
    samp = pl.col("q").hash(SEED + 1) % 6 == 0
    tr, va = [], []
    for pp in parts:
        f = labelled(pp).filter(samp)
        tr.append(f.filter(~pl.col("is_val")))
        va.append(f.filter(pl.col("is_val") & pl.col("half_a")))
    tr, va = pl.concat(tr), pl.concat(va)
    log(f"train rows {tr.height} pos {tr['y'].sum()} | early-stop rows {va.height}")
    # Stage A: small fast model used only to prune obviously wrong candidates.
    model_a = lgb.train(PARAMS_A, lgb.Dataset(_to_x(tr), tr["y"].to_numpy().astype(np.int8), feature_name=feat_names(tr)),
                        num_boost_round=150)
    model_a.save_model(str(work / "modelA.txt"))
    pos_before = int(tr["y"].sum())
    tr = tr.with_columns(pl.Series("pA", model_a.predict(_to_x(tr), num_threads=NPROC).astype(np.float32))).filter(pl.col("pA") >= PRUNE_P)
    va = va.with_columns(pl.Series("pA", model_a.predict(_to_x(va), num_threads=NPROC).astype(np.float32))).filter(pl.col("pA") >= PRUNE_P)
    log(f"stage A pruning: train rows kept {tr.height}, positives kept {tr['y'].sum()}/{pos_before}")
    dtr = lgb.Dataset(_to_x(tr), tr["y"].to_numpy().astype(np.int8), feature_name=feat_names(tr), free_raw_data=True)
    dva = lgb.Dataset(_to_x(va), va["y"].to_numpy().astype(np.int8), reference=dtr, free_raw_data=True)
    evals: dict = {}
    model = lgb.train(PARAMS_B, dtr, num_boost_round=max_rounds, valid_sets=[dtr, dva], valid_names=["train", "valA"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(evals),
                                 lgb.log_evaluation(100)])
    model.save_model(str(work / "model.txt"), num_iteration=model.best_iteration)
    bi = model.best_iteration
    log(f"model trained: best_iteration={bi} train_logloss={evals['train']['binary_logloss'][bi - 1]:.5f} "
        f"valA_logloss={evals['valA']['binary_logloss'][bi - 1]:.5f}")
    del tr, va, dtr, dva

    # Score only queries that can reach an evaluated S1 (val S1 or a train-S1 sample for the gap check).
    s1_all = lab.filter(pl.col("src") == 1)
    s1_val_a = s1_all.filter(pl.col("is_val") & pl.col("half_a"))["rid"].to_list()
    s1_val_b = s1_all.filter(pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
    s1_trs = s1_all.filter(~pl.col("is_val") & (pl.col("rid").hash(SEED + 9) % 40 == 0))["rid"].to_list()
    evalset = pl.Series(s1_val_a + s1_val_b + s1_trs)
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    bests, pos, pos_raw = [], [], []
    for pp in parts:
        f = labelled(pp)
        qs = f.filter(pl.col("s1").is_in(evalset.implode())).select("q").unique()
        f = f.join(qs, on="q")
        pos_raw.append(f.filter(pl.col("y") & pl.col("s1").is_in(evalset.implode())).select("q", "s1"))
        f = _cascade(f, model_a, model)
        bests.append(_best_per_query(f).filter(pl.col("s1").is_in(evalset.implode())))
        pos.append(f.filter(pl.col("y") & pl.col("s1").is_in(evalset.implode())).select("q", "s1"))
        log(f"scored {pp.name}")
    best = pl.concat(bests)
    best.write_parquet(work / "val_best.parquet")
    pos = pl.concat(pos).unique()
    val_all = s1_val_a + s1_val_b
    n_true = sum(len(truth.get(s, ())) for s in val_all)
    pos_val = pos.filter(pl.col("s1").is_in(pl.Series(val_all).implode()))
    pos_raw_val = pl.concat(pos_raw).unique().filter(pl.col("s1").is_in(pl.Series(val_all).implode()))
    rep = {"best_iteration": bi, "val_s1": len(val_all), "true_pairs": n_true,
           "retrieval_recall": pos_raw_val.height / max(n_true, 1),
           "candidate_recall": pos_val.height / max(n_true, 1),
           "oracle_f05": macro_on(val_all, truth, pos_val.iter_rows()), "empty_f05": macro_on(val_all, truth, [])}
    log(json.dumps(rep))
    top_a, ef_a = _eval_block("valA", s1_val_a, truth, best, rep)
    cfg = ({"mode": "efopt", "miss": 0.0, "thr": top_a[1]} if ef_a > top_a[0] else {"mode": "thr", "thr": top_a[1]})
    rep["valB_f05_at_valA_choice"] = _score_at(s1_val_b, truth, best, cfg)
    rep["val_f05_at_valA_choice"] = _score_at(val_all, truth, best, cfg)
    rep["train_sample_f05_at_valA_choice"] = _score_at(s1_trs, truth, best, cfg)
    cfg["f05"] = rep["valB_f05_at_valA_choice"]
    rep["best"] = cfg
    _eval_block("valB_diag", s1_val_b, truth, best, rep)
    imp = dict(zip(model.feature_name(), model.feature_importance("gain").round().tolist()))
    rep["importance"] = dict(sorted(imp.items(), key=lambda x: -x[1]))
    (work / "val_report.json").write_text(json.dumps(rep, indent=1))
    log(f"VAL choice={cfg} | valB={rep['valB_f05_at_valA_choice']:.5f} val={rep['val_f05_at_valA_choice']:.5f} "
        f"train_sample={rep['train_sample_f05_at_valA_choice']:.5f} (gap = overfit indicator)")


def predict(work: Path, out: Path, thr: float | None) -> None:
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(work / "model.txt"))
    model_a = lgb.Booster(model_file=str(work / "modelA.txt"))
    cfg = json.loads((work / "val_report.json").read_text())["best"]
    if thr is not None:
        cfg = {"mode": "thr", "thr": thr}
    rec = pl.read_parquet(work / "test" / "records.parquet", columns=["entity_id", "src"]).with_row_index("rid")
    bests, cands = [], []
    for pp in _parts(work, "test"):
        f = _cascade(pl.read_parquet(pp), model_a, model)  # candidates = pairs the final model scores
        bests.append(_best_per_query(f))
        cands.append(f.select("q", "s1"))
        log(f"predicted {pp.name}")
    ids = rec.select("rid", "entity_id")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))

    def agg(pairs: pl.DataFrame, col: str) -> pl.DataFrame:
        g = (pairs.join(ids.rename({"rid": "q", "entity_id": "m"}), on="q")
                  .group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col)))
        return s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col)

    out.mkdir(parents=True, exist_ok=True)
    best = pl.concat(bests)
    best.write_parquet(work / "test_best.parquet")
    if cfg["mode"] == "efopt":
        chosen = decide_pairs(best, cfg["miss"])
    else:
        chosen = best.filter(pl.col("prob") >= cfg["thr"]).select("q", "s1")
    m = agg(chosen, "matched_entity_ids")
    m.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    agg(pl.concat(cands), "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"predict: wrote {out} cfg={cfg} nonempty={int((m['matched_entity_ids'] != '').sum())}/{m.height}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "retrieve", "add_neural", "features", "features_inc", "train", "train_a",
                                           "predict", "all"])
    ap.add_argument("--split", default="both")
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--work", type=Path, default=Path("/work/run"))
    ap.add_argument("--out", type=Path, default=Path("/work/output"))
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    splits = ["train", "test"] if a.split == "both" else [a.split]
    st = a.stage
    for sp in splits:
        if st in ("prep", "all"):
            prep(a.data, a.work, sp)
        if st in ("retrieve", "all"):
            retrieve(a.work, sp)
        if st == "add_neural":
            add_neural(a.work, sp)
        if st == "features_inc":
            features_incremental(a.work, sp)
        if st in ("features", "all"):
            features(a.work, sp)
    if st in ("train", "all"):
        train(a.data, a.work)
    if st == "train_a":
        train_a(a.data, a.work)
    if st in ("predict", "all"):
        predict(a.work, a.out, a.thr)


if __name__ == "__main__":
    main()
