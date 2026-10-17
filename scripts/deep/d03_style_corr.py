"""Deep dive 3: do copies of the same S1 share perturbation styles (a per-S1 'style' the generator might carry)?
For pairs of copies of one S1, lift = P(style_b | style_a) / P(style_b | not style_a). Lift ~1 -> independent.

uv run --no-project --with polars==1.44.2 --with numpy python scripts/deep/d03_style_corr.py
"""
import re
import unicodedata

import numpy as np
import polars as pl

S = {s: pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)}
tp = pl.read_parquet("local_data/train_pairs.parquet")
s1ids = S[1].sample(250_000, seed=5)["entity_id"]
tp = tp.filter(pl.col("s1").is_in(s1ids.implode()))
rec = pl.concat([S[2], S[3]]).rename({"entity_id": "m", "business_name": "cn", "business_address": "ca"})
x = tp.join(rec, on="m").join(S[1].rename({"entity_id": "s1", "business_name": "sn", "business_address": "sa"}).drop("country"), on="s1")
print("pairs", x.height)


def key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return " ".join(re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch))))


def feats(cn, sn, ca, sa):
    cn, sn, ca, sa = cn or "", sn or "", ca or "", sa or ""
    tc, ts = key(cn).split(), key(sn).split()
    return (
        cn.isupper() and not sn.isupper(),
        cn.islower() and not sn.islower(),
        "  " in cn and "  " not in sn,
        bool(re.search(r"[\[\(\{]", cn)) and not re.search(r"[\[\(\{]", sn),
        sum(ord(c) > 127 for c in cn) > sum(ord(c) > 127 for c in sn),
        bool(re.match(r"^\W", cn)) and not re.match(r"^\W", sn),
        key(cn) == key(sn),
        sorted(tc) == sorted(ts) and tc != ts,
        len(tc) < len(ts),
        len(tc) > len(ts),
        ca.strip() == "",
        ca.isupper(),
        key(ca) == key(sa),
        ca.count(",") != sa.count(","),
    )


NAMES = ["upper", "lower", "dspace", "bracket", "accent", "prefix", "key_eq", "reorder", "drop_tok", "add_tok",
         "addr_empty", "addr_upper", "addr_key_eq", "addr_commas"]
F = np.array([feats(*r) for r in x.select("cn", "sn", "ca", "sa").iter_rows()], dtype=bool)
x = x.with_columns(pl.Series("src", x["m"].str.slice(0, 2)))
s1 = x["s1"].to_numpy(); src = x["src"].to_numpy()
order = np.argsort(s1, kind="stable"); s1o = s1[order]
starts = np.r_[0, np.nonzero(s1o[1:] != s1o[:-1])[0] + 1, len(s1o)]
A, B, SAME = [], [], []
for i in range(len(starts) - 1):
    g = order[starts[i]:starts[i + 1]]
    for a in range(len(g)):
        for b in range(a + 1, len(g)):
            A.append(g[a]); B.append(g[b]); SAME.append(src[g[a]] == src[g[b]])
A, B, SAME = np.array(A), np.array(B), np.array(SAME)
print("copy pairs", len(A))
print(f"{'style':12s} {'base':>7s} {'P(b|a)':>7s} {'P(b|~a)':>7s} {'lift':>6s}  same-src lift / cross-src lift")
for j, n in enumerate(NAMES):
    fa, fb = np.r_[F[A, j], F[B, j]], np.r_[F[B, j], F[A, j]]
    sm = np.r_[SAME, SAME]
    def lift(msk):
        p1 = fb[msk & fa].mean() if (msk & fa).any() else np.nan
        p0 = fb[msk & ~fa].mean()
        return p1, p0, p1 / p0 if p0 > 0 else np.nan
    p1, p0, l = lift(np.ones_like(fa))
    print(f"{n:12s} {F[:, j].mean():7.4f} {p1:7.4f} {p0:7.4f} {l:6.2f}  {lift(sm)[2]:6.2f} / {lift(~sm)[2]:6.2f}")
