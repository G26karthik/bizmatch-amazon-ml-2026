# How our solution came together (team Minutes)

This note tells the story behind our submission for the Amazon ML Challenge 2026: what we built, in what order, and
why each step earned its place. The methodology itself is described in [`APPROACH.md`](APPROACH.md), and the
repository [`README.md`](../README.md) explains how to reproduce every step.

## The problem in brief

Three sources of business records (name, address, country) share no identifiers. Source 1 is a clean reference list;
Sources 2 and 3 contain noisy copies of those businesses together with decoys. For every Source 1 business we list its
copies, and the score is F0.5 averaged per business, so a wrong match costs about three times as much as a missed one.
Training covers the US and India, while the test set adds France without any labels.

## Stage 1: the right formulation (0.921 to 0.965)

Our first observation set the direction for everything that followed: every Source 2 or Source 3 record belongs to at
most one Source 1 business, across all 7.64 million training pairs. We therefore resolve each record to its business
and group the answers. A single TF-IDF search with one LightGBM reached 0.921 on the public leaderboard, and three
pooled searches with a pruning cascade reached 0.951.

We then studied the test set carefully and noticed that it contains more unmatched records per business than the
training data (5.8 against 4.7). We reshaped our validation set to reproduce that density, and from then on every
choice we made was measured on data that behaves like the test set. The score rose to 0.965.

## Stage 2: the modelling stack (0.965 to 0.985628)

With a reliable validation set in place we built the full model:

- a fine-tuned multilingual bi-encoder that adds candidates the text searches miss, bringing recall to 98.9% with
  about five candidates per business;
- a three-stage LightGBM cascade whose later stages see how strongly other records claim the same business;
- seven multilingual cross-encoders, trained on Kaggle and Modal GPUs strictly on records that never touch a validation
  business, to re-read the uncertain pairs;
- a cross-fitted LightGBM stacker and an exact expected-F0.5 optimiser that chooses each business's matches.

Leaderboard probes confirmed the decision strictness chosen on validation. This stage took us to 0.985628.

## Stage 3: France without labels (0.985628 to 0.989496)

A diagnostic submission showed that our US and India predictions scored 0.9913 on the test set, exactly matching
validation, and that France offered the most room to improve. With no French labels, we studied how the generator
builds French records. It creates *sibling* decoys at a business's own address by swapping its distinctive word for
another real French word ("*X* Club SARL" becomes "*X* Amicale SARL"), while genuine copies only misspell words or
swap them for generic ones such as Fils, Cie or Associés.

We confirmed this with two independent label-free tests. The generator's copy caps make genuine copies disappear at
businesses that are already full, while decoys keep appearing there, and a formatting signature (the changed word in
brackets) appears on 4.2% of copies but only 0.9% of decoys. The rules built on this evidence added 0.0039 to our
score, in line with our prediction, and brought us to 0.989496.

## Stage 4: calibrating for the test set (0.989496 to 0.990162)

Next we examined what the test set's extra records really are. Empty-address records, which are almost always copies,
occur at the same rate per business in training and test. That told us the extra test records are decoys with
addresses rather than copies of businesses missing from Source 1. We retrained the final calibration step with this in
mind, adding generator-aware features, label-free orphan features and two Qwen2.5-7B judges fine-tuned with LoRA on
Lightning AI GPUs, together with structural rules for one-token names such as initials and web handles. This produced
our strongest leaderboard run at 0.990162, matching the gain we had predicted from validation.

## Stage 5: the final submission

For the final submission we extended that run with 3,744 further matches: structural rules (unique names, one-token
names at an address, transliterated Indic names) and learned tie-breakers for ambiguous empty-address records. We
applied one strict selection criterion. Each addition had to be a record that the calibrated model had not already
scored, so that every new match adds information instead of overriding an existing judgement. The resulting file,
v28q_fr48, was our final submission.

## What we are most proud of

- A validation design that tracked the test set closely enough that validation and test agreed to four decimal places
  for the US and India.
- Precise rules for a country with no labels, built entirely from the structure of the data generator and verified
  with statistical tests.
- Near-complete candidate recall at about five candidates per business.
- A disciplined workflow: every change was verified with a paired bootstrap on a held-out half before it reached the
  leaderboard.

## Compute and tools

We had no sponsored or institutional compute, so we built the whole solution on free tiers, promotional credits and
one of our own laptops: AWS EC2 for the pipeline and stack, Kaggle (2×T4) and a Modal H100 for the cross-encoders,
Lightning AI (RTX PRO 6000) for the LoRA judges, and a laptop RTX 4060 for the bi-encoder and experiments. Those limits
kept us disciplined: every stage saves its results so it can resume on another machine, and every GPU job is a small
script that can run wherever free capacity was available that day. All models are licensed under MIT or Apache-2.0
with at most 8 billion parameters, and we used only the provided data.
