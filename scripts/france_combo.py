"""Combined France candidate (US / India decisions unchanged; production decision rule everywhere).

  1. base decisions from --base_tag (v28 stack + cross-encoders, strictness 1.5)
  2. optional: France decisions replaced by those of a France-fixed blend (--france_tag in --france_work: stack features
     recomputed from France-cleaned records, original cross-encoder scores)
  3. gate v1: France pairs the France-cleaned cross-encoder blend (--clean_tag) selects that carry the true-copy signature
     (house number equal, street similarity >= 90) on France-fixed features, for records not yet assigned
  4. gate v2: unassigned France records at their best S1's exact address whose name keeps a distinctive word of the S1 name
     (type-word swaps such as "X Pharmacie" / "X Sportive"), no decoy word, S1 under the per-source copy cap
  5. optional additions from a France-fixed blend (--add_tag): its France pairs for unassigned records whose house number
     is equal or missing (France-fixed parsing), no decoy word
  6. optional --mhr: remove France pairs whose house numbers (France-fixed parsing) differ by 1-50 with no parse-artefact
     marker and one number per address: generator decoys move the house number by 1-50 (clipped at 1: a spike of
     records at house "1"), while French copies almost never carry house-number noise (deleted-digit pairs 602 vs 74,560 US)
Records that add a decoy word (holding, participations, international, distribution, groupe, developpement, france) are
never added by 3 or 4: those words mark generator decoys, which the France-cleaned text strips.

python scripts/france_combo.py --work /work/run28 --base_tag run28_6ce --clean_tag run28_6ce_frce \
    --france_work /work/runfr28 --france_tag fx6o --pruned /work/runfr/test/pruned --out /work/out_combo
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _chosen, _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--base_tag", required=True)
ap.add_argument("--clean_tag", required=True)
ap.add_argument("--france_work", type=Path)
ap.add_argument("--france_tag")
ap.add_argument("--pruned", type=Path, required=True, help="France pair features used for the signature (France-fixed)")
ap.add_argument("--add_work", type=Path)
ap.add_argument("--add_tag")
ap.add_argument("--fixed_records", type=Path, default=Path("/work/runfr/test/records.parquet"))
ap.add_argument("--mhr", action="store_true")
ap.add_argument("--cap_trim", action="store_true")
ap.add_argument("--extra_pairs", type=Path, help="parquet (q, s1) of rule-based additions (e.g. initials copies at the S1 address)")
ap.add_argument("--nonempty_only", action="store_true", help="France additions only for S1s the base decisions already match")
ap.add_argument("--novel_names", action="store_true", help="v2 / fixed-blend additions only for records whose name is no other France S1's name")
ap.add_argument("--no_v1", action="store_true")
ap.add_argument("--no_v2", action="store_true")
ap.add_argument("--max_df", type=float, default=0.002)
ap.add_argument("--out", type=Path)
a = ap.parse_args()

STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e"}
DECOY = {"holding", "participations", "international", "distribution", "groupe", "developpement", "france"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", s)) - STOP


def decide(work, tag):
    cfg = json.loads((work / f"val_report_{tag}.json").read_text())["best"]
    best = pl.read_parquet(work / f"test_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    return best, _chosen(best, cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))


rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country", "business_name"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
fr_s1 = rec.filter(pl.col("country") == "France").select(pl.col("rid").alias("s1"))
names = rec.select("rid", "business_name")
best, cur = decide(a.work, a.base_tag)
print(f"base decisions {cur.height} (France {cur.join(fr_s1, on='s1', how='semi').height})")
if a.france_tag:
    fbest, fch = decide(a.france_work, a.france_tag)
    fch = fch.join(fr_s1, on="s1", how="semi")
    cur = pl.concat([cur.join(fr_s1, on="s1", how="anti"), fch])
    best = pl.concat([best.join(fr_s1, on="s1", how="anti"), fbest.join(fr_s1, on="s1", how="semi")])
    print(f"France decisions from {a.france_tag}: {fch.height}")
feat = pl.concat([pl.read_parquet(p, columns=["q", "s1", "street_tsr", "house_eq"]) for p in sorted(a.pruned.glob("France*.parquet"))]) \
         .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
sig = feat.filter(pl.col("house_eq").fill_null(False) & (pl.col("street_tsr") >= 90)).select("q", "s1")


def no_decoy(x):
    x = x.join(names.rename({"rid": "q", "business_name": "nq"}), on="q").join(names.rename({"rid": "s1", "business_name": "ns"}), on="s1")
    keep = [not ((toks(p) - toks(s)) & DECOY) for p, s in zip(x["nq"].to_list(), x["ns"].to_list())]
    return x.filter(pl.Series(keep)).select("q", "s1")


def nkey(s):
    return " ".join(sorted(toks(s)))


# a record whose name is exactly another France S1's name may be a copy of that S1 (changed address), not a name-noised
# copy of the S1 at its address: keep such records out of the forced additions (v2 and fixed-blend additions)
S1KEYS = {}
for r, n in rec.filter((pl.col("src") == 1) & (pl.col("country") == "France")).select("rid", "business_name").iter_rows():
    S1KEYS.setdefault(nkey(n), set()).add(r)


def novel_name(x):
    x = x.join(names.rename({"rid": "q", "business_name": "nq"}), on="q")
    keep = [not (S1KEYS.get(nkey(n), set()) - {s}) for n, s in zip(x["nq"].to_list(), x["s1"].to_list())] if a.novel_names else [True] * x.height
    return x.filter(pl.Series(keep, dtype=pl.Boolean)).select("q", "s1")


# S1s the base decisions already match: the generator leaves 5.58% of S1s without copies in every country, and France
# predictions sit below that rate, so additions to S1s the models left empty mostly hit true singletons (score 0)
BASE_NONEMPTY = cur.select("s1").unique()


def only_nonempty(x):
    return x.join(BASE_NONEMPTY, on="s1", how="semi") if a.nonempty_only else x


if not a.no_v1:
    _, clean = decide(a.work, a.clean_tag)
    add1 = clean.join(fr_s1, on="s1", how="semi").join(cur, on=["q", "s1"], how="anti").join(sig, on=["q", "s1"], how="semi") \
                .join(cur.select("q"), on="q", how="anti")
    # same-house pairs keep decoy words here (as in the leaderboard-tested gate): with the house unchanged they are copies
    add1 = only_nonempty(add1)
    cur = pl.concat([cur, add1])
    print(f"gate v1 (cleaned blend, signature): +{add1.height}")
if not a.no_v2:
    c = best.join(fr_s1, on="s1", how="semi").join(sig, on=["q", "s1"], how="semi").join(cur.select("q"), on="q", how="anti") \
            .join(names.rename({"rid": "q", "business_name": "nq"}), on="q").join(names.rename({"rid": "s1", "business_name": "ns"}), on="s1") \
            .join(rec.select(pl.col("rid").alias("q"), "src"), on="q")
    s1names = rec.filter((pl.col("src") == 1) & (pl.col("country") == "France"))["business_name"].to_list()
    df = Counter(t for n in s1names for t in toks(n))
    cut = a.max_df * len(s1names)
    keep = []
    for nq, ns in zip(c["nq"].to_list(), c["ns"].to_list()):
        tq, ts = toks(nq), toks(ns)
        keep.append(not ((tq - ts) & DECOY) and any(df.get(t, 0) <= cut for t in tq & ts))
    c = c.filter(pl.Series(keep))
    cnt = cur.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
    c = c.join(cnt, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0))
    add2 = novel_name(c.filter(pl.when(pl.col("src") == 2).then(pl.col("n2") < 5).otherwise(pl.col("n3") < 6)).select("q", "s1"))
    add2 = only_nonempty(add2)
    cur = pl.concat([cur, add2])
    print(f"gate v2 (same address, type-word swap): +{add2.height}")
fx = pl.read_parquet(a.fixed_records, columns=["house", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
hq = fx.select(pl.col("rid").alias("q"), pl.col("house").alias("hq"), pl.col("business_address").alias("aq"))
hs = fx.select(pl.col("rid").alias("s1"), pl.col("house").alias("hs"), pl.col("business_address").alias("as_"))
if a.add_tag:
    _, ach = decide(a.add_work, a.add_tag)
    add3 = ach.join(fr_s1, on="s1", how="semi").join(cur.select("q"), on="q", how="anti").join(hq, on="q").join(hs, on="s1")
    add3 = add3.filter(pl.col("hq").is_null() | pl.col("hs").is_null() | (pl.col("hq") == "") | (pl.col("hs") == "") | (pl.col("hq") == pl.col("hs")))
    add3 = novel_name(no_decoy(add3.select("q", "s1")))
    add3 = only_nonempty(add3)
    cur = pl.concat([cur, add3])
    print(f"France-fixed blend additions ({a.add_tag}, house equal or missing, no decoy word): +{add3.height}")
if a.mhr:
    ART = re.compile(r"(?i)(appart|\bapt\b|\bapp\b|\bbat\b|b[aâ]t\.|b[aâ]timent|porte|\bpte\b|[ée]tage|escalier|\besc\b|r[ée]sidence|\blot\b|lotissement|"
                     r"\d+\s*(bis|ter|quater)\b|\d+(bis|ter|b|t)\b|cedex|\bbp\b|\bcs\b|\bzi\b|\bza\b|\bzac\b|\bkm\b|\bcour\b|\bbox\b|\bcase\b)")
    NUM = re.compile(r"\d+")

    def one_num(x):
        return len([n for n in NUM.findall(x or "") if len(n) != 5]) == 1

    # generator decoys: house = S1 house + one of these offsets, never lower (train and test, US and France)
    DECOY_OFFSETS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
    m = cur.join(fr_s1, on="s1", how="semi").join(hq, on="q").join(hs, on="s1")
    drop = []
    for q, s1, h1, h2, a1, a2 in zip(*[m[c].to_list() for c in ("q", "s1", "hq", "hs", "aq", "as_")]):
        if h1 and h2 and h1 != h2 and h1.isdigit() and h2.isdigit() and (int(h1) - int(h2)) in DECOY_OFFSETS \
                and not (ART.search(a1 or "") or ART.search(a2 or "")) and one_num(a1) and one_num(a2):
            drop.append((q, s1))
    drop = pl.DataFrame(drop, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")
    cur = cur.join(drop, on=["q", "s1"], how="anti")
    print(f"decoy-offset removal (France, house = S1 + 1,2,3,4,5,7,9,11,13,21, no artefact): -{drop.height}")
if a.extra_pairs:  # validated rule additions (initials copies: 176/176 true on validation), unassigned records only
    ex = pl.read_parquet(a.extra_pairs).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).unique(subset="q", keep="none")
    ex = ex.join(cur.select("q"), on="q", how="anti")
    cur = pl.concat([cur, ex])
    print(f"extra rule pairs ({a.extra_pairs.name}): +{ex.height}")
if a.cap_trim:  # generator caps: at most 5 S2 and 6 S3 copies per S1; keep the most probable ones
    pr = best.select("q", "s1", "prob")
    x = cur.join(pr, on=["q", "s1"], how="left").with_columns(pl.col("prob").fill_null(0.5)) \
           .join(rec.select(pl.col("rid").alias("q"), "src"), on="q") \
           .with_columns(pl.col("prob").rank("ordinal", descending=True).over("s1", "src").alias("_r"))
    over = x.filter(((pl.col("src") == 2) & (pl.col("_r") > 5)) | ((pl.col("src") == 3) & (pl.col("_r") > 6))).select("q", "s1")
    cur = cur.join(over, on=["q", "s1"], how="anti")
    print(f"cap trim: -{over.height}")
assert cur.select("q").is_duplicated().sum() == 0, "a record was assigned twice"
print(f"final pairs {cur.height} (France {cur.join(fr_s1, on='s1', how='semi').height})")
if a.out:
    cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
    _write_outputs(a.work, a.out, {"mode": "thr", "thr": 0.5}, cur.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob")),
                   cands, rec.select("rid", "entity_id", "src"))
