"""Deep dive 5: can sibling copies break namesake ties? Train truth. Namesake group = S1s of one country with the same
normalized name key (2..40 members). Tie record = empty-address S2/S3 true copy of a group member. For each candidate
namesake: its other copies with a non-empty address (what a matcher finds) -> counts and style agreement with the tie
record. Conditional-logit LightGBM (binary on candidate rows, normalized within record), trained on half the groups,
evaluated on the other half: top-1 accuracy and share of records with p_top >= 0.8, count-only vs count+style.

uv run --no-project --with polars==1.44.2 --with numpy --with lightgbm --with rapidfuzz python scripts/deep/d05_tie_siblings.py
"""
import re
import unicodedata
import zlib
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein


def key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return " ".join(re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch))))


def style(cn, sn):
    tc, ts = key(cn).split(), key(sn).split()
    return np.array([cn.isupper() and not sn.isupper(), cn.islower() and not sn.islower(), "  " in cn,
                     bool(re.search(r"[\[\(\{]", cn)), sum(ord(c) > 127 for c in cn) > sum(ord(c) > 127 for c in sn),
                     bool(re.match(r"^\W", cn)), key(cn) == key(sn), sorted(tc) == sorted(ts) and tc != ts,
                     len(tc) < len(ts), len(tc) > len(ts), cn == sn, cn.strip() != cn, "," in cn and "," not in sn,
                     "." in cn and "." not in sn, "-" in cn and "-" not in sn, "&" in cn and "&" not in sn], dtype=np.int8)


NS = 16
S = {s: pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)}
tp = pl.read_parquet("local_data/train_pairs.parquet")
s1 = S[1].with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
grp = s1.group_by("country", "k").agg(pl.col("entity_id").alias("ids")).filter(pl.col("ids").list.len().is_between(2, 40))
gid = {e: i for i, ids in enumerate(grp["ids"].to_list()) for e in ids}
members = grp["ids"].to_list()
s1name = dict(zip(s1["entity_id"], s1["business_name"]))
rec = pl.concat([S[2], S[3]]).rename({"entity_id": "m"})
x = tp.filter(pl.col("s1").is_in(list(gid))).join(rec, on="m")
cop = defaultdict(list)   # s1 -> list of (m, src, name, empty)
for s, m, n, a in x.select("s1", "m", "business_name", "business_address").iter_rows():
    cop[s].append((m, m[:2], n or "", (a or "").strip() == ""))
print("groups", grp.height, "S1s in groups", len(gid), "copies of group members", x.height)
rows, lab, rid, grp_of = [], [], [], []
r_i = 0
for s, lst in cop.items():
    for (m, src, n, empty) in lst:
        if not empty:
            continue
        g = gid[s]
        sr = style(n, s1name[s])
        for c in members[g]:
            sib = [(cn, csrc) for (cm, csrc, cn, ce) in cop.get(c, []) if cm != m and not ce]
            same = [cn for cn, csrc in sib if csrc == src]; oth = [cn for cn, csrc in sib if csrc != src]
            f = [len(same), len(oth), len(members[g])]
            for L in (same, oth):
                if L:
                    st = np.array([style(cn, s1name[c]) for cn in L])
                    f += list((st == sr).mean(0)); f += [max(1 - Levenshtein.normalized_distance(n, cn) for cn in L),
                                                         np.mean([1 - Levenshtein.normalized_distance(n, cn) for cn in L]),
                                                         float(any(cn == n for cn in L))]
                else:
                    f += [np.nan] * (NS + 3)
            rows.append(f); lab.append(int(c == s)); rid.append(r_i); grp_of.append(g)
        r_i += 1
X = np.array(rows, dtype=np.float32); y = np.array(lab); rid = np.array(rid); grp_of = np.array(grp_of)
print("tie records", r_i, "candidate rows", len(y))
half = np.array([zlib.crc32(str(g).encode()) % 2 for g in grp_of])


def evaluate(cols, name):
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=1)
    tr, te = half == 0, half == 1
    b = lgb.train(P, lgb.Dataset(X[tr][:, cols], y[tr]), 400)
    s = b.predict(X[te][:, cols], raw_score=True)
    r, yt = rid[te], y[te]
    df = pl.DataFrame({"r": r, "s": s, "y": yt}).with_columns((pl.col("s") - pl.col("s").max().over("r")).exp().alias("e"))
    df = df.with_columns((pl.col("e") / pl.col("e").sum().over("r")).alias("p"))
    top = df.sort("p", descending=True).group_by("r").first()
    for th in (0.5, 0.7, 0.8, 0.9):
        t = top.filter(pl.col("p") >= th)
        print(f"  {name:12s} p_top>={th}: share {t.height / top.height:.4f} acc {t['y'].mean() if t.height else float('nan'):.4f}")
    print(f"  {name:12s} top-1 acc {top['y'].mean():.4f}, mean p_top {top['p'].mean():.4f}, n records {top.height}")


evaluate([0, 1, 2], "count-only")
evaluate(list(range(X.shape[1])), "count+style")
