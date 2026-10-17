"""Exact leaderboard metric: macro F0.5 over Source 1 entities (singletons included)."""
from __future__ import annotations

from typing import Iterable, Mapping


def entity_f05(true_ids: Iterable[str], predicted_ids: Iterable[str]) -> float:
    t, p = set(true_ids), set(predicted_ids)
    if not t:
        return 1.0 if not p else 0.0
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def macro_f05(truth: Mapping[str, Iterable[str]], pred: Mapping[str, Iterable[str]]) -> float:
    """Average over every S1 id in `truth`; ids missing from `pred` count as empty predictions."""
    if not truth:
        raise ValueError("empty truth")
    return sum(entity_f05(t, pred.get(s1, ())) for s1, t in truth.items()) / len(truth)


def parse_id_list(field: str) -> list[str]:
    return [x for x in (s.strip() for s in field.split(",")) if x]
