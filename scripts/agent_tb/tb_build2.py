"""Agent TB step 1: candidate sets for AMBIGUOUS empty-address records (>=2 S1s one word-op away), features per pair.

Candidate generation follows scripts/deep/d38_empty_oneop.py (exact / minus indexes, ops add/drop/swap/typo) but keeps
ALL candidates. Writes local_data/work/tb_pairs_{train,test}.parquet.

uv run --no-project --with polars==1.44.2 --with rapidfuzz --with lightgbm --with numpy python scripts/agent_tb/tb_build.py
"""
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
GEN = {"center", "centre", "services", "service", "group", "trading", "enterprises", "enterprise", "partners", "solutions",
       "solution", "international", "consulting", "consultants", "associates", "holdings", "industries", "systems",
       "technologies", "technology", "tech", "global", "store", "stores", "shop", "restaurant", "traders", "agency",
       "management", "corporation", "products", "marketing", "sales", "studio", "clinic", "hospital", "school", "hotel",
       "cafe", "bar", "salon", "ventures", "llc", "inc", "india", "america", "usa", "us", "new", "home", "care", "health",
       "medical", "the", "and", "of", "properties", "realty", "construction", "motors", "auto", "foods", "food"}
MAXC = 12      # candidate sets larger than this are skipped (not resolvable at high precision; avoids blow-up)
OPS = {"add": 0, "drop": 1, "typo": 2, "swap": 3}
CTRY = {"US": 0, "India": 1, "France": 2}
ACC = {"à": "a", "á": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ç": "c", "è": "e", "é": "e", "ê": "e", "ë": "e",
       "ì": "i", "í": "i", "î": "i", "ï": "i", "ñ": "n", "ò": "o", "ó": "o", "ô": "o", "ö": "o", "õ": "o", "ù": "u",
       "ú": "u", "û": "u", "ü": "u", "ý": "y", "ÿ": "y", "œ": "oe", "æ": "ae", "ß": "ss"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def key(ts):
    return " ".join(sorted(ts))


def nkey_expr(col="business_name"):
    """Vectorised approximation of key(set(toks(name)) - LEG) for name-count features."""
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_many(list(ACC), list(ACC.values()))
            .str.extract_all(r"\w+").list.eval(pl.element().filter(~pl.element().is_in(list(LEG))))
            .list.unique().list.sort().list.join(" "))


def run(split, keep=None, truth=None):
    t0 = time.time()
    s1 = pl.read_parquet(f"local_data/{split}_source1.parquet", columns=["entity_id", "business_name", "country"])
    emp = (pl.col("business_address").fill_null("").str.strip_chars() == "")
    oth = pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet").filter(emp)
                     .select("entity_id", "business_name", "country").collect() for s in (2, 3)])
    if truth is not None:
        tq = truth.filter(pl.col("m").is_in(oth["entity_id"].implode()))
        truth = dict(zip(tq["m"].to_list(), tq["s1"].to_list()))
    print(split, "loaded", round(time.time() - t0), "s", flush=True)
    exact, minus = defaultdict(list), defaultdict(list)
    df = Counter()
    for e, n, c in s1.iter_rows():
        tl = toks(n)
        T = set(tl) - LEG
        for t in T:
            df[(c, t)] += 1
        if not T:
            continue
        exact[(c, key(T))].append(e)
        if len(T) >= 2:
            for t in T:
                minus[(c, key(T - {t}))].append((e, t))
    ns1 = Counter(s1["country"].to_list())
    print(split, "S1 index", round(time.time() - t0), "s", flush=True)
    del s1
    q = oth
    if keep is not None:
        q = q.filter(pl.col("entity_id").is_in(keep))
    rows, amb = [], []
    nrec = nbig = 0
    for i, (m, n, c) in enumerate(q.select("entity_id", "business_name", "country").iter_rows()):
        if i % 50000 == 0:
            print(split, "progress", i, "/", q.height, "ambiguous", nrec, "pairs", len(rows), round(time.time() - t0), "s", flush=True)
        rtl = toks(n)
        R = set(rtl) - LEG
        if not R or exact.get((c, key(R))):
            continue
        cand = {}
        if len(R) >= 2:
            for r in R:
                for e in exact.get((c, key(R - {r})), ()):
                    cand.setdefault(e, ("add", r, None))
        for e, t in minus.get((c, key(R)), ()):
            cand.setdefault(e, ("drop", None, t))
        big = False
        for r in R:
            lst = minus.get((c, key(R - {r})), ())
            if len(lst) > MAXC + 1:
                big = True
                break
            for e, t in lst:
                if t != r:
                    cand.setdefault(e, ("typo" if Levenshtein.distance(r, t) <= max(1, min(len(r), len(t)) // 4) else "swap", r, t))
        if big or len(cand) > MAXC:
            nbig += 1
            continue
        if len(cand) < 2:
            continue
        nrec += 1
        amb.append((m, n, c, rtl, R, cand))
    print(split, "ambiguous records", nrec, f"skipped >{MAXC} candidates: {nbig}", round(time.time() - t0), "s", flush=True)
    del q, oth
    need = {e for a_ in amb for e in a_[5]}
    s1n = (pl.scan_parquet(f"local_data/{split}_source1.parquet").filter(pl.col("entity_id").is_in(list(need)))
           .select("entity_id", "business_name", "business_address").collect())
    s1info = {e: (n, a, toks(n)) for e, n, a in s1n.iter_rows()}
    keys = {(a_[2], key(a_[4])) for a_ in amb} | {(c_, key(set(v[2]) - LEG)) for a_ in amb for c_ in [a_[2]] for e in a_[5] for v in [s1info[e]]}
    kdf = pl.DataFrame({"country": [k[0] for k in keys], "nk": [k[1] for k in keys]})
    parts = []
    for s in (2, 3):
        parts.append(pl.scan_parquet(f"local_data/{split}_source{s}.parquet")
                     .select("country", nk=nkey_expr(), emp=(pl.col("business_address").fill_null("").str.strip_chars() == ""))
                     .join(kdf.lazy(), on=["country", "nk"], how="semi")
                     .group_by("country", "nk").agg(n=pl.len(), n_emp=pl.col("emp").sum()).with_columns(src=pl.lit(f"S{s}"))
                     .collect(engine="streaming"))
    NKC = {}
    for c_, k_, n_, ne_, s_ in pl.concat(parts).iter_rows():
        a0 = NKC.get((c_, k_), [0, 0, 0, 0])
        a0[0] += n_; a0[1 if s_ == "S2" else 2] += n_; a0[3] += ne_
        NKC[(c_, k_)] = a0
    print(split, "name counts", len(NKC), round(time.time() - t0), "s", flush=True)
    for m, n, c, rtl, R, cand in amb:
        tr = truth.get(m) if truth is not None else None
        ops = Counter(v[0] for v in cand.values())
        rk = key(R)
        rc = NKC.get((c, rk), (0, 0, 0, 0))
        src = m[:2]
        rnorm = " ".join(rtl)
        rleg = set(rtl) & LEG
        N = ns1[c]
        for e, (op, r, t) in cand.items():
            sn, sa, stl = s1info[e]
            ST = set(stl) - LEG
            sk = key(ST)
            cc = NKC.get((c, sk), (0, 0, 0, 0))
            atoks = set(toks(sa)) if sa else set()
            chg = r if op == "add" else t          # the word present on one side only
            df_r = df.get((c, r), 0) if r else -1
            df_t = df.get((c, t), 0) if t else -1
            snorm = " ".join(stl)
            pos_t = (stl.index(t) / max(1, len(stl) - 1)) if t and t in stl else -1
            pos_r = (rtl.index(r) / max(1, len(rtl) - 1)) if r and r in rtl else -1
            rows.append(dict(
                m=m, s1=e, country=c, kind=op, label=(tr == e) if truth is not None else None,
                has_truth=(tr is not None) if truth is not None else None,
                op_id=OPS[op], ctry=CTRY[c], src=int(src[1]), ncand=len(cand),
                n_add=ops["add"], n_drop=ops["drop"], n_typo=ops["typo"], n_swap=ops["swap"],
                ntok_r=len(R), ntok_s=len(ST),
                len_r=len(r) if r else -1, len_t=len(t) if t else -1,
                ldf_r=math.log1p(df_r) if df_r >= 0 else -1.0, ldf_t=math.log1p(df_t) if df_t >= 0 else -1.0,
                ldf_chg=math.log1p(df.get((c, chg), 0)), frac_chg=df.get((c, chg), 0) / N,
                gen_r=int(r in GEN) if r else -1, gen_t=int(t in GEN) if t else -1,
                dig_r=int(r.isdigit()) if r else -1, dig_t=int(t.isdigit()) if t else -1,
                lev=Levenshtein.distance(r, t) if (r and t) else -1,
                levn=Levenshtein.normalized_distance(r, t) if (r and t) else -1.0,
                pref=int(r.startswith(t) or t.startswith(r)) if (r and t) else -1,
                fc_same=int(r[0] == t[0]) if (r and t) else -1,
                s1_dup=len(exact.get((c, sk), ())),
                c_all=cc[0], c_same=cc[1] if src == "S2" else cc[2], c_oth=cc[2] if src == "S2" else cc[1], c_emp=cc[3],
                r_dup=rc[0] - 1, r_emp=rc[3] - 1,
                ratio=fuzz.ratio(rnorm, snorm), tsr=fuzz.token_set_ratio(rnorm, snorm), tsort=fuzz.token_sort_ratio(rnorm, snorm),
                pratio=fuzz.partial_ratio(rnorm, snorm),
                leg_eq=int(rleg == (set(stl) & LEG)), leg_n_r=len(rleg), leg_n_s=len(set(stl) & LEG),
                first_eq=int(bool(rtl) and bool(stl) and rtl[0] == stl[0]), order_eq=int([x for x in rtl if x in ST] == [x for x in stl if x in R]),
                s_addr_emp=int(not atoks), chg_in_addr=int(chg in atoks), pos_t=pos_t, pos_r=pos_r,
                len_raw_r=len(n or ""), len_raw_s=len(sn or ""), case_eq=int((n or "") == (sn or "")),
                case_ieq_loose=int((n or "").lower().replace(" ", "") == (sn or "").lower().replace(" ", "")),
            ))
    print(split, "pairs", len(rows), round(time.time() - t0), "s", flush=True)
    return pl.DataFrame(rows, infer_schema_length=None)


if __name__ == "__main__":
    which = sys.argv[1:] or ["train", "test"]
    if "train" in which:
        tp = pl.read_parquet("local_data/train_pairs.parquet")
        tr = run("train", truth=tp)
        tr.write_parquet("local_data/work/tb_pairs_train.parquet")
    if "test" in which:
        g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t",
                        schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
        asg = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").select(m="matched_entity_ids", fr27_s1="source1_entity_id"))
        te = run("test")   # all empty-address ambiguous records; fr27 status joined (apply only to unassigned)
        te = te.drop("label", "has_truth").join(asg, on="m", how="left")
        te.write_parquet("local_data/work/tb_pairs_test.parquet")
