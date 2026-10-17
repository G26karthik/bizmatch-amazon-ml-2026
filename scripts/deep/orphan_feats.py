"""Label-free orphan-cluster features for uncertain pairs (q record, s1 candidate), val (train split) and test.
For key k in {nkey(q), nkey(s1)} (country-local; nkey = sorted name tokens minus legal forms/stopwords):
  vis_*   : visible... all S1s in the file with key k (test: all S1s; val: S1s not hidden)
  far_*   : other S2/S3 records with key k, non-empty address, address token-Jaccard with the S1's address < 0.5
            (evidence of another, possibly hidden, entity with this name)
  nearq_* : of those, how many have address Jaccard >= 0.5 with q's address (a cluster at q's own address)
  near_*  : other records with key k near the S1's address (the S1's likely copies)
  empty_* : other empty-address records with key k (competing tie records)
Writes local_data/llm/orphan_{val,test}.parquet (q, s1, features). Groups larger than 400 records get NaN.

uv run --no-project --with polars==1.44.2 --with numpy python scripts/deep/orphan_feats.py
"""
import re
import unicodedata
from collections import defaultdict

import numpy as np
import polars as pl

SEED, HIDE, CAP = 42, 19, 400
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a", "de", "du", "des", "la", "le", "les", "et"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def nkey(s):
    return " ".join(sorted(set(toks(s)) - LEG))


def run(split, unc_path, out):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
    if split == "train":
        rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
        rec = rec.filter(~pl.col("hidden"))
    unc = pl.read_parquet(unc_path).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).unique()
    rec = rec.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"))
    ctry = dict(zip(rec["rid"].to_list(), rec["country"].to_list()))
    kof = dict(zip(rec["rid"].to_list(), rec["k"].to_list()))
    need = set()
    for q, s in unc.iter_rows():
        need.add((ctry.get(q), kof.get(q))); need.add((ctry.get(s), kof.get(s)))
    sub = rec.filter(pl.struct("country", "k").map_elements(lambda r: (r["country"], r["k"]) in need, return_dtype=pl.Boolean))
    print(split, "records in needed keys", sub.height, flush=True)
    grp = defaultdict(list)
    atok = {}
    for rid, c, k, src, a in sub.select("rid", "country", "k", "src", "business_address").iter_rows():
        grp[(c, k)].append((rid, src))
        atok[rid] = frozenset(toks(a))
    jac = lambda a, b: len(a & b) / len(a | b) if a and b else 0.0  # noqa: E731
    feats = []
    for q, s in unc.iter_rows():
        c = ctry[q]; aq, as_ = atok.get(q, frozenset()), atok.get(s, frozenset())
        row = [q, s]
        for kk in (kof[q], kof[s]):
            g = grp.get((c, kk), [])
            if len(g) > CAP:
                row += [float(sum(1 for _, src in g if src == 1))] + [np.nan] * 4
                continue
            vis = far = nearq = near = emp = 0
            for rid, src in g:
                if rid in (q, s):
                    continue
                if src == 1:
                    vis += 1; continue
                a = atok[rid]
                if not a:
                    emp += 1; continue
                if jac(a, as_) >= 0.5:
                    near += 1
                else:
                    far += 1
                    if aq and jac(a, aq) >= 0.5:
                        nearq += 1
            row += [vis, far, nearq, near, emp]
        feats.append(row)
    names = ["q", "s1"] + [f"{p}_{w}" for w in ("kq", "ks") for p in ("vis", "far", "nearq", "near", "empty")]
    df = pl.DataFrame(feats, schema=names, orient="row").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    df.write_parquet(out)
    print("wrote", out, df.height, flush=True)
    print(df.describe())


run("train", "local_data/llm/unc_val.parquet", "local_data/llm/orphan_val.parquet")
run("test", "local_data/llm/unc_test.parquet", "local_data/llm/orphan_test.parquet")
