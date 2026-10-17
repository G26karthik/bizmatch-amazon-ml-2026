"""agent_mo task A: EMPTY-ADDRESS records whose nkey (sorted tokens minus legal/stop words) is exactly TWO word-operations
from exactly one S1 of the country, and NOT within one op of any S1 (and not an exact nkey match).
Set edit distance: dist(R,T) = max(|R\\T|, |T\\R|) (add/drop/swap-or-typo each 1 op). Core R&T must be non-empty.
Op pair labels: add+add (2,0), drop+drop (0,2), add+swap (2,1), drop+swap (1,2), swap+swap (2,2); a swap is 'typo' when
the replaced word is within Levenshtein max(1, len//4); 'merge' when a record word = concatenation of two dropped S1 words
(or a split, the reverse).
Train (all S1 visible) precision by class; test: all records in class (assigned in fr27 to same S1 / other / unassigned).
Implied test precision among unassigned = (train copies per S1 * test S1s - fr27 accepted same) / unassigned.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_mo/a_empty_twoop.py
"""
import re
import unicodedata
from collections import defaultdict
from itertools import combinations

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
OUT = "local_data/work/agent_mo_A_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def key(ts):
    return " ".join(sorted(ts))


def subkeys(T, k=2):
    T = sorted(T)
    out = []
    for r in range(0, min(k, len(T) - 1) + 1):   # keep core non-empty
        for drop in combinations(T, r):
            out.append((key(set(T) - set(drop)), drop))
    return out


def istypo(r, t):
    return Levenshtein.distance(r, t) <= max(1, min(len(r), len(t)) // 4)


def label(X, Y):
    """X = record-only words, Y = S1-only words."""
    a, d = len(X), len(Y)
    X, Y = sorted(X), sorted(Y)
    if a == 1 and d == 2 and (X[0] == Y[0] + Y[1] or X[0] == Y[1] + Y[0]):
        return "merge"
    if a == 2 and d == 1 and (Y[0] == X[0] + X[1] or Y[0] == X[1] + X[0]):
        return "split"
    if (a, d) == (2, 0):
        return "add+add"
    if (a, d) == (0, 2):
        return "drop+drop"
    if (a, d) == (2, 1):
        return "add+" + ("typo" if any(istypo(x, Y[0]) for x in X) else "swap")
    if (a, d) == (1, 2):
        return "drop+" + ("typo" if any(istypo(X[0], y) for y in Y) else "swap")
    if (a, d) == (2, 2):
        nt = max(istypo(X[0], Y[0]) + istypo(X[1], Y[1]), istypo(X[0], Y[1]) + istypo(X[1], Y[0]))
        return ["swap+swap", "typo+swap", "typo+typo"][nt]
    return f"{a},{d}"


def run(split):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "country"]) for s in (1, 2, 3)])
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1")).select("entity_id", "business_name", "country")
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() == "")).select("entity_id", "business_name", "country")
    del rec
    qrows = []
    need = set()
    for m, n, c in q.iter_rows():
        R = set(toks(n)) - LEG
        if not R:
            continue
        qrows.append((m, c, R))
        for k_, _ in subkeys(R):
            need.add((c, k_))
    idx = defaultdict(list)
    for e, n, c in s1.iter_rows():
        T = set(toks(n)) - LEG
        if not T:
            continue
        tk = key(T)
        for k_, _ in subkeys(T):
            if (c, k_) in need:
                idx[(c, k_)].append((e, tk))
    del need
    res = []
    for m, c, R in qrows:
        cand = {}
        for k_, _ in subkeys(R):
            for e, tk in idx.get((c, k_), ()):
                if e in cand:
                    continue
                T = set(tk.split(" "))
                cand[e] = (max(len(R - T), len(T - R)), R - T, T - R)
        if not cand:
            continue
        dmin = min(v[0] for v in cand.values())
        if dmin <= 1:
            continue
        two = [(e, v) for e, v in cand.items() if v[0] == 2]
        if len(two) != 1:
            continue
        e, (_, X, Y) = two[0]
        res.append((c, m, e, label(X, Y), len(R), len(R & (R ^ X)), max(len(t) for t in R)))
    return pl.DataFrame(res, schema={"country": pl.String, "m": pl.String, "s1": pl.String, "op": pl.String, "ntok": pl.Int64,
                                    "ncore": pl.Int64, "maxlen": pl.Int64}, orient="row"), s1.group_by("country").len().rename({"len": "S1s"})


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr, n1tr = run("train")
tr = tr.join(tp.rename({"s1": "s1_true"}), on="m", how="left").with_columns((pl.col("s1_true") == pl.col("s1")).fill_null(False).alias("right"))
tr.write_parquet(OUT.format("train"))
print("TRAIN (all visible): empty-address, two ops from exactly one S1, none within one op")
print(tr.group_by("country", "op").agg(pl.len(), pl.col("right").sum().alias("n_right"), pl.col("right").mean().round(4).alias("P")).sort("country", "op"))
print(tr.group_by("country", "op", pl.col("ncore").clip(1, 3)).agg(pl.len(), pl.col("right").mean().round(4).alias("P")).sort("country", "op", "ncore"))

te, n1te = run("test")
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
a = a.rename({"source1_entity_id": "s1_fr27", "matched_entity_ids": "m"})
te = te.join(a, on="m", how="left").with_columns(
    pl.when(pl.col("s1_fr27").is_null()).then(pl.lit("unassigned")).when(pl.col("s1_fr27") == pl.col("s1")).then(pl.lit("acc_same")).otherwise(pl.lit("acc_other")).alias("status"))
te.write_parquet(OUT.format("test"))

for gb in (["country", "op"], ["country", "op", "ncore"]):
    trg = tr.with_columns(pl.col("ncore").clip(1, 3)).group_by(gb).agg(pl.len().alias("tr_n"), pl.col("right").sum().alias("tr_right")).join(n1tr, on="country")
    teg = te.with_columns(pl.col("ncore").clip(1, 3)).group_by(gb).agg(pl.len().alias("te_n"), (pl.col("status") == "acc_same").sum().alias("te_acc"),
                                                                   (pl.col("status") == "acc_other").sum().alias("te_oth"), (pl.col("status") == "unassigned").sum().alias("te_un")).join(n1te.rename({"S1s": "S1te"}), on="country")
    x = trg.join(teg, on=gb, how="full", coalesce=True).fill_null(0).with_columns(
        (pl.col("tr_right") / pl.col("tr_n")).round(4).alias("P_train"),
        (pl.col("tr_n") / pl.col("S1s") * 1000).round(3).alias("tr_all_1k"),
        (pl.col("tr_right") / pl.col("S1s") * 1000).round(3).alias("tr_copy_1k"),
        (pl.col("te_n") / pl.col("S1te") * 1000).round(3).alias("te_all_1k"),
        (pl.col("te_acc") / pl.col("S1te") * 1000).round(3).alias("te_acc_1k"),
        (pl.col("te_un") / pl.col("S1te") * 1000).round(3).alias("te_un_1k"))
    x = x.with_columns((pl.col("te_all_1k") / (pl.col("tr_all_1k") + 1e-9)).round(3).alias("all_ratio"),
                       ((pl.col("tr_copy_1k") - pl.col("te_acc_1k")) / (pl.col("te_un_1k") + 1e-9)).round(3).alias("P_implied_un"))
    print(x.drop("S1s", "S1te").sort(gb))
