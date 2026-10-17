"""Score a prediction file on the team's validation split (Amazon ML Challenge 2026, business entity resolution).

Standard library only (Python 3.9+). Usage:

    python score_val.py --pred my_val_predictions.tsv \
        --gt student_resource/dataset/train/train_ground_truth.tsv --split val_split.csv.gz [--half B]

--pred: TSV with header `source1_entity_id<TAB>matched_entity_ids` (same format as matching_results.tsv) containing
        predictions for the validation S1 entities of the chosen half. Missing S1 rows count as empty predictions;
        rows for other S1 entities are ignored.
--split: val_split.csv.gz from the kit (entity_id,role), roles: valA_s1, valB_s1, hidden_s1, val_record.
--half: B (default) for the reported score, A while tuning thresholds, AB for both halves together.

Metric: macro F0.5 over S1 entities (the leaderboard metric). "hard" additionally counts every false match that comes
from an unmatched record (a record with no true S1 once hidden S1s are removed) 3 times, because the test set has more
decoy-like unmatched records than training.
"""
import argparse
import csv
import gzip
import sys

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def entity_f05(t: set, p: set, extra_fp: int = 0) -> float:
    if not t:
        return 1.0 if not p else 0.0
    tp = len(t & p)
    fp = len(p - t) + extra_fp
    fn = len(t - p)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def read_pairs(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        if [h.strip() for h in header[:2]] not in (["source1_entity_id", "matched_entity_ids"],
                                                    ["source1_entity_id", "candidate_entity_ids"]):
            sys.exit(f"{path}: unexpected header {header}")
        for row in r:
            if not row:
                continue
            s1 = row[0].strip()
            ids = [x.strip() for x in (row[1] if len(row) > 1 else "").split(",") if x.strip()]
            if s1 in out:
                sys.exit(f"{path}: duplicate row for {s1}")
            out[s1] = ids
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True, help="train_ground_truth.tsv")
    ap.add_argument("--split", required=True, help="val_split.csv.gz")
    ap.add_argument("--half", default="B", choices=["A", "B", "AB"])
    ap.add_argument("--hard_k", type=int, default=3)
    a = ap.parse_args()

    roles = {}
    with gzip.open(a.split, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roles[row["entity_id"]] = row["role"]
    want = {"A": {"valA_s1"}, "B": {"valB_s1"}, "AB": {"valA_s1", "valB_s1"}}[a.half]
    eval_s1 = [e for e, r in roles.items() if r in want]
    hidden = {e for e, r in roles.items() if r == "hidden_s1"}

    gt = read_pairs(a.gt)
    truth, matched = {}, set()
    for s1, ids in gt.items():
        if s1 in hidden:  # hidden S1 entities are not in the pool: their records count as unmatched
            continue
        truth[s1] = set(ids)
        matched.update(ids)
    pred = read_pairs(a.pred)

    owner = {}
    conflicts = 0
    for s1 in eval_s1:
        for q in pred.get(s1, ()):
            if q in owner and owner[q] != s1:
                conflicts += 1
            owner[q] = s1
    bad_s1 = sum(1 for s1 in pred if roles.get(s1) == "hidden_s1" and pred[s1])
    plain, hard, sing, sing_ok = 0.0, 0.0, 0, 0
    n_fp = n_fp_orphan = n_tp = n_fn = 0
    for s1 in eval_s1:
        t, p = truth.get(s1, set()), set(pred.get(s1, ()))
        orphan_fp = sum(1 for q in p - t if q not in matched)
        plain += entity_f05(t, p)
        hard += entity_f05(t, p, (a.hard_k - 1) * orphan_fp)
        n_tp += len(t & p); n_fp += len(p - t); n_fp_orphan += orphan_fp; n_fn += len(t - p)
        if not t:
            sing += 1
            sing_ok += not p
    n = len(eval_s1)
    covered = sum(1 for s1 in eval_s1 if s1 in pred)
    print(f"half {a.half}: {n} S1 entities ({covered} have a row in the prediction file)")
    print(f"macro F0.5 plain = {plain / n:.5f}")
    print(f"macro F0.5 hard (orphan false matches x{a.hard_k}) = {hard / n:.5f}")
    print(f"pairs: true positives {n_tp}, false positives {n_fp} (from unmatched records {n_fp_orphan}), missed {n_fn}")
    print(f"singletons: {sing}, correctly left empty {sing_ok}")
    if conflicts:
        print(f"warning: {conflicts} records are assigned to more than one S1 (each record matches at most one S1)")
    if bad_s1:
        print(f"warning: {bad_s1} hidden S1 entities have predictions; hidden S1s must be excluded from the pool")


if __name__ == "__main__":
    main()
