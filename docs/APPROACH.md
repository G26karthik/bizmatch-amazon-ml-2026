# Approach: Business Entity Resolution (Amazon ML Challenge 2026, Team Minutes)

**Team Name:** Minutes  
**Team Members:** Gowrawaram Karthik Koundinya (team leader), Macha Jayanth, M V Abhiram, Hasanabada Sirichandana  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We approached entity resolution from the record's side. Every Source 2 and Source 3 record in the training data
belongs to at most one Source 1 business, so for each record we ask a single question: which business does it belong
to, if any? Grouping those answers by business gives the final matches. Candidates come from three TF-IDF searches and
a fine-tuned multilingual bi-encoder, which keep 98.75% of true matches with only 5.4 candidates per business. A
three-stage LightGBM cascade ranks them while taking into account how strongly other records claim the same business,
seven fine-tuned multilingual cross-encoders re-read the uncertain pairs, and a cross-fitted LightGBM stacker combines
every signal. An exact expected-F0.5 optimiser then selects the best set of matches for each business.

Two ideas carried the most weight. We built a validation set that behaves like the test set, first in how many
unmatched records it contains and then in what kind of records they are, and we tuned every decision on it. For
France, which has no training labels, we modelled the data generator directly and confirmed each rule with two
independent label-free statistical tests before using it. Our final submission builds on our strongest leaderboard run
(public score 0.990162) and adds 3,744 high-precision matches.

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale and structure.** The training set has 2,206,821 Source 1 businesses, 10,320,219 Source 2 and Source 3 records
and 7,638,365 true pairs. Three properties of the data shaped the whole design.

* **Every record matches at most one business**, and the country always agrees. Matching can therefore run country by
  country and record by record.
* **The generator caps copies**: at most 5 from Source 2 and 6 from Source 3, with a mean of 3.46 per business. 5.6% of
  businesses have no copies at all, and the metric rewards an empty prediction for them.
* **About a quarter of training records match nothing**, and the test set has more of them: 5.5 to 5.8 records per
  business instead of 4.7.

**Noise patterns we catalogued.**

* **Names:** changes of case and accents, brackets ("[Inc]"), decorative prefixes ("--", "<<"), legal-form variants
  (Inc, Incorporated, Pvt Ltd, Private Limited, SARL), reordered words, typos, an added generic word ("Center",
  "Services"), aliases ("X d/b/a NAME", "trading as"), one-token forms such as initials, web handles
  ("brandname.com") and invented brand names, and names written in Devanagari, Bengali, Telugu, Tamil or Kannada.
* **Addresses:** abbreviations (Rd and Road, St and Street, R. and Rue), reordered components, missing PIN codes or
  states, landmark references, house-number typos, "bis" and "ter" suffixes, and 3.4% of records with no address.
  Empty-address records are copies 97.7% of the time, so they deserve special care.

**Decoys.** Non-matching records are designed to look plausible: near-duplicate names ("*Name* Diner" and "*Name*
Diner Southside"), the same name at a different address and, in France, *sibling* records at a business's own address
in which its distinctive word is replaced by another real French word ("*X* Club SARL" becomes "*X* Amicale SARL").

**How the test set differs.** France appears only in the test set (259,452 of 1,732,544 businesses). The test set also
contains more unmatched records per business. We studied their nature closely: empty-address records per business are
identical in training and test (US 0.168 and 0.1665, India 0.139 and 0.138), which shows that the extra test records
are decoys with addresses rather than copies of businesses missing from Source 1. This finding guided our final
calibration step.

**The hardest cases.** The largest remaining source of error is empty-address records whose name is shared by several
businesses (one generic company name is shared by 38 businesses). We resolve these where the data contains
evidence, such as the spelling details a copy keeps from its original.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-retriever blocking, a cascaded gradient-boosted classifier with collective (group)
features, cross-encoder re-scoring of uncertain pairs, a stacked blend and an exact expected-F0.5 decision rule,
followed by generator-aware rules for France and a test-like recalibration.

**Core Innovations:**

1. **Record-to-business assignment** with an exact per-business expected-F0.5 decision, which optimises the
   precision-weighted, per-business metric directly.
2. **Test-like validation.** We split training data by business, never by pair, and divide the validation businesses
   into half A (used for every choice) and half B (used only for reporting). The validation set mirrors the test set's
   share and type of unmatched records.
3. **Leak-free cross-encoders**, trained only on records that never touch a validation business, so the stacker learns
   from honest scores.
4. **Label-free rules for France**, each confirmed by two independent markers derived from the generator: its copy caps
   and a formatting signature.
5. **Test-like recalibration** of the uncertain zone with generator-aware features and two LoRA-tuned 7B judges.

### 2.3 Validation Design

Leaderboard uploads were limited, far too few to guide dozens of design choices, so we invested early in a validation
set we could trust.

* **Split by business.** A business joins validation when `hash(entity_id, seed 42) % 10 == 0`, and all of its records
  go with it, so no business is ever split between training and validation. A second hash divides the validation
  businesses into half A, which chooses the number of trees, the stacker and the decision rule, and half B, which is
  used only to report scores.
* **Matching the test density.** The test set has about 5.8 Source 2 and Source 3 records per business against 4.7 in
  training, so roughly 40% of its records match nothing, against 26%. During training and validation we hide 19% of
  Source 1 businesses from the candidate pool. Their records stay in the data but can no longer be matched, which
  reproduces the test set's share of unmatched records.
* **Matching the test difficulty.** Unmatched test records are more decoy-like than the ones that hiding creates, so
  the decision rule is chosen on a stricter version of the metric that counts every false match from an unmatched
  record three times.
* **Matching the kind of unmatched records.** Once we found that the extra test records are decoys with addresses
  rather than copies of missing businesses (Section 2.1), we built a test-like variant of the validation set without
  the records that exist only because of hiding, and used it for the final calibration step.
* **Paired comparisons.** Every change was compared with its predecessor on the same half B businesses with a paired
  bootstrap, which gives a confidence interval for the difference rather than a single number.

This design paid off: a diagnostic submission later measured our US and India predictions at 0.9913 on the test set,
the same value as validation half B.

---

## 3. Candidate Generation (Blocking)

For every Source 2 and Source 3 record, within its country:

- **Normalisation:** Unicode transliteration of Indian scripts, abbreviation maps for the US, India and France, US state
  names, legal-suffix handling, house-number splitting (45BIS becomes 45 bis) and consonant skeletons that make
  transliterations agree ("laaiph investtmentt" and "Life Investment" both become "lf inbstmnt").
- **Blocking keys used:**
  - TF-IDF (hashed, 2^23 features) over address words and adjacent word pairs, and over name words, character 3-grams
    and consonant-skeleton tokens. We pool three searches: name plus address (top 10), address only (top 15) and name
    only (top 8, widened to top 40 when the address is empty).
  - A fine-tuned `intfloat/multilingual-e5-small` bi-encoder over the raw "name | address" text, which reads Indian
    scripts natively, returning the 5 nearest businesses by cosine similarity on GPU.
  - For empty-address records, an additional exact and one-edit name lookup across all businesses of the country.
- **Pruning:** a compact LightGBM (150 trees, 31 leaves) on inexpensive pair features removes candidates below
  probability 0.01. The pruned set is exactly what the downstream models score, and it is the content of
  `candidate_pairs.tsv`.
- **Candidate pairs generated:** about 26 per record before pruning and **9,365,329 test pairs after pruning (5.4 per
  Source 1 business)**.
- **How we ensured true matches were not lost:** we measured recall on held-out businesses after every change. One
  TF-IDF search kept 95.5% of true pairs; all searches together with the bi-encoder keep 98.9%, and 98.75% after
  pruning.

| Pruning cutoff | Candidates per business (train) | True pairs kept |
|---|---|---|
| 0.001 | 6.27 | 98.93% |
| **0.01 (used)** | **4.82** | **98.75%** |
| 0.02 | 4.42 | 98.59% |
| 0.05 | 4.00 | 98.21% |

---

## 4. Matching Model

**Features used (about 50 per pair plus 17 group features):**
- **Name features:** RapidFuzz ratio, token-set ratio and partial ratio, Jaro-Winkler on cleaned names, the same
  measures on consonant skeletons, first-token equality, TF-IDF name cosine, bi-encoder cosine and legal-form relation.
- **Address features:** ratio, token-set and partial ratio on normalised addresses and on street words alone, TF-IDF
  address cosine, house-number equality, digit-level edit similarity and log absolute difference, postcode equality,
  shared-number counts and missing-field flags.
- **Other:** retrieval ranks and score gaps in every search, candidate counts, and **group features**: the out-of-fold
  probability's rank and share within the record, the margin to the runner-up, the probability mass other records
  already place on the same business (overall and per source), the number of confident competing claims and the
  similarity to the strongest competing record. Name-frequency features capture how common a name is.

**Model type:**
- **Three-stage LightGBM cascade.** Stage 1 uses two out-of-fold models (the fold is a hash of the record). Stage 2 adds
  group features computed from out-of-fold stage-1 probabilities, and stage 3 recomputes them from out-of-fold stage-2
  probabilities. Final settings: learning rate 0.02, 383 leaves, feature fraction 0.55, at least 100 records per leaf
  and early stopping on validation half A with patience 250.
- **Seven cross-encoders** fine-tuned as pair classifiers on the uncertain zone (each record's top 5 candidates):
  multilingual-e5-small, two multilingual-e5-base models, multilingual-e5-large, two BAAI/bge-reranker-v2-m3 models and
  Qwen3-Reranker-0.6B. Each was trained on 0.7 to 1.2 million pairs from records that never touch a validation
  business; the best single model reaches a validation log-loss of 0.144.
- **Stacker:** a LightGBM with 15 leaves over the stage-3 logit, every cross-encoder logit, each logit's margin to the
  record's best alternative, cross-encoder mass per business and per record, and raw pair similarities. It is
  cross-fitted on validation half A with two folds by business, which keeps half B untouched.

**Threshold selection method:** instead of one global threshold, we optimise the metric business by business. Each
record is first assigned to its most probable business. For one business, let the records assigned to it have
probabilities p1 ≥ p2 ≥ ... ≥ pn after sharpening (each probability raised to the power γ = 1.5). Selecting the top k
records scores

```
F0.5 = 1.25·TP / (1.25·TP + FP + 0.25·FN),   FP = k - TP
```

where TP follows a Poisson-binomial distribution over p1..pk and FN one over pk+1..pn. An empty prediction scores 1
only when the business truly has no copies, which happens with probability ∏(1 - pi). We compute the **expected F0.5**
of every k exactly with a dynamic programme over both distributions and keep the best k; for independent probabilities
the optimal subset is always such a prefix (Lewis, 1995; Ye et al., ICML 2012). The exponent γ was chosen on half A
from {0.8, 1.0, 1.5, 2.0, 3.0} under the stricter metric, and leaderboard probes on either side of it confirmed 1.5 as
the best setting. We keep at most 5 Source 2 and 6 Source 3 matches per business, in line with the generator's caps.

**France.** France has no training labels, so we modelled its generator directly and confirmed every rule with two
independent label-free tests.
- **Cap test.** Copies are capped per business and source, so a class of genuine copies disappears at businesses whose
  other records already fill the cap, while decoys continue to appear there.
- **Bracket marker.** Copy noise places the changed word in brackets for 4.2% of French copies and only 0.9% of decoys.
  The model's probabilities do not depend on brackets, which makes the marker an independent witness.

Using these tests we remove same-address one-word swaps to another French vocabulary word (sibling decoys, bracket
share 0.8%) while keeping swaps to a misspelled or generic word (bracket share 6.0%, the signature of copies). We also
remove records at the generator's decoy house-number offsets and add generic-word swap copies, address-parse-gap copies
(bis and b, quai and q) and brand-name and initials copies at an address that belongs to a single French business. The
French rules lifted our leaderboard score by 0.0039, in line with the gain the markers predicted.

**Test-like recalibration.** Building on the finding that the test set contains no copies of missing businesses, we
retrained a residual LightGBM on the validation uncertain zone after removing the records that existed only because of
our earlier simulation of missing businesses. Its inputs are the production probability, explicit generator-operation
features (legal-form relation, case, accents, brackets, word swaps, additions and drops, typo similarity, house and
street relation), label-free orphan features (how many businesses share the name and how many same-name records lie
far from the business's address) and the mean logit of two Qwen2.5-7B judges fine-tuned with LoRA on 400,000 training
pairs each. On test-like validation half B it raises macro F0.5 by **+0.00037 (95% CI [+0.00025, +0.00050])** and
stays positive when validation decoys are doubled to test density. The recalibrated probabilities feed the same
decision rule for US and India.

**Structural additions.** Some copies can be recognised from structure alone. We measured each rule on training truth
with every business visible, applied it only to records that were still unassigned and respected the generator's caps.
- One-token names (brand names, initials, web handles) at the address of exactly one business: 98% to 100% correct.
- Initials and web handles of a business on the same street when the house number is missing or mistyped: 95% to 100%.
- Empty-address records whose name identifies exactly one business, or is a single word-operation away from exactly
  one business: 96% to 99%.
- Indic-script names transliterated with a word dictionary learned from training pairs and matched at the same unit
  number: 99%.

**Final submission.** The submitted file adds 3,744 of these structural and learned
matches to our strongest leaderboard run (public score 0.990162). We selected them with one criterion: the record had
not already been scored by the calibrated model, so each addition brings new information rather than overriding an
existing judgement. Learned tie-breakers for ambiguous empty-address records and namesake groups contribute to this
set; they reach 88% to 98% precision on held-out training data in the confidence bands we used.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.991316** on validation half B for the stacked model, with the decision rule chosen on
  half A (0.990823 under a stricter evaluation that counts every false match from an unmatched record three times).
  Test-like recalibration adds a further +0.00037 on test-like half B.
- **Public leaderboard progression:** 0.921, 0.951, 0.965 (test-like validation), 0.984, 0.9855, 0.985628 (seven
  cross-encoders), 0.989496 (French rules) and **0.990162** (test-like recalibration with structural additions). The
  final submission is this run plus 3,744 high-precision additions.
- **Final output at a glance:** 5,866,818 matches over 1,732,544 businesses (3.39 per business, against 3.46 true
  copies per business in training), with 99,324 businesses (5.7%) left empty, against the 5.6% of training businesses
  that have no copies. Every match lies inside the 9,365,329 pairs of `candidate_pairs.tsv`, and both output files pass the
  official validator with `--check-ids`.
- **Consistency between validation and test:** a diagnostic submission measured US and India at 0.9913 on the test set,
  equal to validation half B, which confirms that the validation design generalises. France rose from 0.953 to about
  0.979 with the generator-aware rules.
- **Precision check:** for every class of name change and address relation, the number of accepted pairs per business
  on the test set matches the number of true copies per business in training (ratios 0.99 to 1.02), so the accepted set shows no
  sign of systematic false matches.
- **Common false positives (wrong merges):** decoys at the same building or street, branches of one brand, and
  empty-address records with generic names.
- **Common false negatives (missed matches):** empty-address records whose name is shared by several businesses,
  heavily altered names combined with changed house numbers, and transliterated names with little address overlap.
- **Rigour:** every change we adopted was verified with a paired bootstrap over the 89,000 businesses of validation
  half B before it reached the leaderboard.

---

## 6. Conclusion

Record-to-business assignment, multi-retriever blocking and a cascaded gradient-boosted model with group features
resolve the great majority of matches, and seven cross-encoders together with an exact expected-F0.5 decision handle
the uncertain remainder. Three practices made the difference: a validation set that mirrors the test set in both the
amount and the nature of its unmatched records, label-free markers that let us engineer precise rules for a country
without labels, and a disciplined final assembly that adds only matches carrying new evidence.

The lesson we take from this challenge is that understanding how the data was made paid off more than any single
model. After the core model was in place, our largest gains came from questions about the data itself: how many
unmatched records the test set holds, what those records look like, and how the generator writes a French copy. We
built everything on free compute, which taught us to keep each stage lean, resumable and easy to verify.

---

## Appendix

### A. Code

See the repository layout and run commands in the [README](../README.md). Entry points (from the repository
root with `PYTHONPATH=src`): `python -m ber.pipeline {prep,retrieve,add_neural,features,train_a}`,
`python -m ber.neural {train,encode,retrieve,score}`, `python -m ber.stage2 {prune,export,train_s3}` and
`python -m ber.blend {dump,blend}`, followed by the post-processing scripts in `scripts/`.

### B. Additional Results

| Version | Change | Validation (half B) | Public LB |
|---|---|---|---|
| v1 | one TF-IDF search, 29 features, one LightGBM | 0.936 | 0.921 |
| v2 | three searches, 46 features, pruning cascade | 0.963 | 0.951 |
| v4 | test-like validation, bi-encoder, sibling features | 0.980 | 0.965 |
| v19 | leak-free e5 cross-encoders, LightGBM stacker | 0.9900 | |
| v22 | retrained stack, wider cross-encoder zone | 0.9903 | 0.984 |
| v26 | bge-reranker and a second e5-base cross-encoder | 0.9909 | 0.9855 |
| fr7 | larger stack, seven cross-encoders, decoy offsets, initials copies | 0.991316 | 0.985628 |
| fr14 | generator-aware French rules | 0.991316 | 0.989496 |
| fr22 | test-like recalibration and structural additions | +0.00037 (test-like) | **0.990162** |
| **fr48 (final submission)** | fr22 plus 3,744 high-precision additions | | builds on 0.990162 |

### C. Models, Licences and Compute

| Model | Licence | Parameters | Role |
|---|---|---|---|
| LightGBM | MIT | | pruning, stack, stacker, recalibration |
| intfloat/multilingual-e5-small | MIT | 118M | bi-encoder and cross-encoder |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoders |
| intfloat/multilingual-e5-large | MIT | 560M | cross-encoder |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 568M | cross-encoders |
| Qwen/Qwen3-Reranker-0.6B | Apache-2.0 | 0.6B | cross-encoder |
| Qwen/Qwen2.5-7B and Qwen2.5-7B-Instruct (LoRA) | Apache-2.0 | 7.6B | judges in the recalibration |

Every model is licensed under MIT or Apache-2.0 and has at most 8 billion parameters. We used only the provided training
and test files, with no external data, APIs or geocoding.

**Compute.** We had no sponsored or institutional compute. The whole solution ran on free tiers and promotional credits
and on one of our own laptops.

| Resource | Used for |
|---|---|
| AWS EC2 r7i.xlarge and r7i.2xlarge (promotional credits) | data preparation, retrieval, features, the three-stage stack, stacker and decision rule |
| Kaggle 2×T4 (free quota) | cross-encoder training and scoring |
| Modal H100 (free credits) | cross-encoder training |
| Lightning AI RTX PRO 6000 (free credits) | the two Qwen2.5-7B LoRA judges and cross-encoder jobs |
| Laptop RTX 4060, 8 GB | bi-encoder fine-tuning, encoding and nearest-neighbour search, experiments |

Working within free limits shaped the engineering. Every stage writes Parquet artefacts so that it can resume on
another machine, long CPU jobs ran partly on spot instances under a watchdog that relaunches the machine and resumes
from checkpoints, and each GPU job is a small script that uploads its inputs and returns only scores.
