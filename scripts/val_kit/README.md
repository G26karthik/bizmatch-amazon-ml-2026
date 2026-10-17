# Team validation kit — Amazon ML Challenge 2026 (business entity resolution)

Score your model locally on exactly the validation we use, before sending anything for the leaderboard.
It needs only Python 3.9+ (standard library) and the training data you already have
(`student_resource/dataset/train/`).

## Files

| File | What it is |
|---|---|
| `score_val.py` | The scorer: macro F0.5 over S1 entities (the leaderboard metric), plain and "hard" |
| `val_split.csv.gz` | `entity_id,role` for training entities: `valA_s1`, `valB_s1`, `hidden_s1`, `val_record` |
| `our_val_predictions_v22.tsv` | Our v22 predictions for the validation S1 entities (to check the scorer and compare) |

## The protocol (follow all five rules, or the score means nothing)

1. **Hide 19% of S1 entities.** Remove every `hidden_s1` entity from the S1 side, in training *and* validation. Their
   S2/S3 records then have no true match. This reproduces the test set, where about 40% of S2/S3 records match
   nothing (vs 26% in raw training data). Skipping this rule inflates your score and hides decoy problems.
2. **Never train on validation.** Exclude the `valA_s1` and `valB_s1` entities and every `val_record` (their S2/S3
   records) from anything supervised: model fitting, feature selection, early stopping, threshold search on half B.
   Unsupervised steps over all text (TF-IDF vocabulary, normalisation) are fine.
3. **Predict against the full training pool.** For each validation S1 entity, search *all* training S2/S3 records
   (training-side records included; they act as decoys), with the S1 pool = all training S1 entities minus `hidden_s1`.
   Each S2/S3 record may belong to at most one S1.
4. **Tune on half A, report half B.** Choose thresholds and settings with `--half A`; report only `--half B`.
5. **Same file format as the submission:** TSV header `source1_entity_id<TAB>matched_entity_ids`, one row per
   validation S1 entity (half A and B), matched ids comma-separated, empty when none.

## Run

```bash
python score_val.py --pred my_val_predictions.tsv \
    --gt student_resource/dataset/train/train_ground_truth.tsv --split val_split.csv.gz --half B
```

Check the scorer first on our file: `--pred our_val_predictions_v22.tsv --half B` must print
`plain = 0.99027` and `hard = 0.98957`.

## Reference numbers (half B → public leaderboard)

| Version | plain | hard | Leaderboard |
|---|---|---|---|
| v4 | 0.980 | ≈0.976 | 0.965 |
| v22 (current best) | 0.99027 | 0.98957 | 0.984 |

"hard" counts each false match coming from an unmatched record 3 times; the leaderboard has been between our hard
score and about 0.011 below it. A file that does not beat **0.9896 hard** on half B will not improve our submission.
Also send the half-B output of `score_val.py` along with any test file you share.

## Prompt you can give your AI coding assistant

> I am working on the Amazon ML Challenge 2026 entity-resolution task (match Source 2/3 business records to Source 1
> entities; metric = macro F0.5 per S1 entity). Our team validates with a fixed kit: `val_split.csv.gz`
> (entity_id, role in {valA_s1, valB_s1, hidden_s1, val_record}) and `score_val.py`. Adapt my pipeline to follow these
> rules exactly: (1) drop every hidden_s1 entity from the S1 pool in training and validation, so their records become
> unmatched; (2) exclude valA_s1, valB_s1 and val_record entities from all supervised training, feature selection and
> early stopping; (3) for every valA_s1 and valB_s1 entity, predict matches by searching all training S2/S3 records
> against the pool of all non-hidden training S1 entities, each record assigned to at most one S1; (4) tune thresholds
> only with `python score_val.py --half A`, and report `--half B`; (5) write predictions as a TSV with header
> `source1_entity_id<TAB>matched_entity_ids`. Then run the scorer and show me the half-B output.
