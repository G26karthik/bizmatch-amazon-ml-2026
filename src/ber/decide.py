"""Per-S1 subset selection maximizing expected F0.5 under independent match probabilities.

For one S1 with candidate probabilities p (sorted desc), the optimal subset under independence is a top-k prefix
(Lewis 1995; Nan Ye et al., ICML 2012 "Optimizing F-measures: A Tale of Two Approaches"). For each k we compute
E[F0.5] exactly via Poisson-binomial distributions of TP (selected) and FN (unselected):
    F = 1.25 TP / (1.25 TP + (k - TP) + 0.25 FN),  and F = 1 when truth and prediction are both empty.
`miss` adds expected true matches outside the candidate set (retrieval misses) as extra FN mass.
"""
from __future__ import annotations

import numpy as np
from numba import njit


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
def _best_k(p, miss):
    n = len(p)
    best_k, best_v = 0, _pb(p)[0] * np.exp(-miss)  # empty prediction scores 1 only if no true match at all
    for k in range(1, n + 1):
        tp = _pb(p[:k])
        fn = _pb(p[k:])
        v = 0.0
        for a in range(k + 1):
            if tp[a] == 0.0:
                continue
            if a == 0:
                continue  # F = 0 when TP = 0 and prediction nonempty
            inner = 0.0
            for b in range(n - k + 1):
                inner += fn[b] * 1.25 * a / (1.25 * a + (k - a) + 0.25 * (b + miss))
            v += tp[a] * inner
        if v > best_v:
            best_v, best_k = v, k
    return best_k, best_v


@njit(cache=True)
def select(group_start, p_sorted, miss):
    """group_start: offsets into p_sorted (per-S1 blocks, each sorted desc). Returns boolean keep mask."""
    keep = np.zeros(len(p_sorted), dtype=np.bool_)
    for g in range(len(group_start) - 1):
        s, e = group_start[g], group_start[g + 1]
        k, _ = _best_k(p_sorted[s:e], miss)
        keep[s:s + k] = True
    return keep
