"""Generator dive 1: profile true pairs (S1 record vs its copy) -- name and address noise, by country and source.

Runs on the worker: python g1_truepairs.py [N]   (needs /work/dive/tp.parquet from g0_samples.py, bnorm.py = ber.normalize)
Writes /work/dive/g1_pairs.parquet (per-pair flags) for later joins.
"""
import collections
import difflib
import re
import sys

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as LV
from unidecode import unidecode

sys.path.insert(0, "/work/dive")
from bnorm import ADDR_ABBR, LEGAL, addr_tokens, name_core, skeleton  # noqa: E402

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
PART, NPART = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
tp = pl.read_parquet("/work/dive/tp_s2.parquet", columns=["s1r", "mr", "msrc", "country", "business_name_1",
                     "business_address_1", "business_name_2", "business_address_2"]).rename(
    {"business_name_1": "n1", "business_address_1": "a1", "business_name_2": "n2", "business_address_2": "a2"})
tp = tp.filter(pl.int_range(pl.len()) % NPART == PART)
print("pairs", tp.height, flush=True)

TOK = re.compile(r"[a-z0-9]+|&")
LEG = set(LEGAL) - {"the", "and", "of", "dba"} | {"pvtltd", "opc", "llc", "pte"}
PHRASE = {"formerly", "aka", "fka", "dba", "com", "also", "known", "as", "trading", "ta", "previously"}
PH_RE = re.compile(r"\b(formerly|fka|f/k/a|aka|a/k/a|dba|d/b/a|also known as|trading as|previously)\b|\.com\b", re.I)
NONLAT = re.compile("[ऀ-෿]")  # Indic scripts (Devanagari..Malayalam)
ACCENT = re.compile("[À-ɏ]")  # accented Latin letters
WEB = re.compile(r"(?i)\.com\b|www\.|@|\|")
HON = {"mr", "mrs", "ms", "smt", "shri", "sri", "dr"}


def ntoks(s):
    return TOK.findall(unidecode(s or "").lower().replace(".", "").replace("'", ""))


def bare(s):
    return re.sub(r"[^a-z0-9]", "", unidecode(s or "").lower())


def casepat(s):
    L = [c for c in (s or "") if c.isalpha()]
    if not L:
        return "n"
    if any(ord(c) > 127 for c in L):
        return "x"
    if all(c.isupper() for c in L):
        return "U"
    if all(c.islower() for c in L):
        return "l"
    ws = re.findall(r"[A-Za-z]+", s)
    if all(w[0].isupper() and (len(w) == 1 or w[1:].islower()) for w in ws):
        return "T"
    return "M"


def punct(s):
    return collections.Counter(c for c in (s or "") if not c.isalnum() and not c.isspace())


def tok_pair(x, y):
    if x in LEG and y in LEG:
        return "legal_swap"
    if {x, y} == {"&", "and"}:
        return "amp_and"
    d = LV.distance(x, y)
    m = max(len(x), len(y))
    if d == 1 and m >= 3 or d == 2 and m >= 6:
        if len(x) == len(y) and sorted(x) == sorted(y):
            return "typo_transpose"
        if len(y) == len(x) - 1:
            return "typo_del"
        if len(y) == len(x) + 1:
            return "typo_ins"
        return "typo_sub" if d == 1 else "typo_2"
    if len(x) >= 1 and len(y) >= 1 and (x.startswith(y) or y.startswith(x)):
        return "abbrev"
    if skeleton(x) == skeleton(y):
        return "phonetic"
    if m >= 5 and LV.normalized_similarity(x, y) >= 0.4:
        return "garble"
    return "word_replace"


def name_ops(n1, n2):
    """multi-label set of operations turning S1 name n1 into copy name n2 + detail words."""
    ops, det = set(), []
    if n1 == n2:
        return {"exact"}, det
    c1, c2 = casepat(n1), casepat(n2)
    if c1 != c2:
        ops.add(f"case:{c1}>{c2}")
    if n1.lower() == n2.lower():
        ops.add("case_only")
        return ops, det
    nl1, nl2 = bool(NONLAT.search(n1)), bool(NONLAT.search(n2))
    if nl2 and not nl1:
        ops.add("translit_to_nonlatin")
        return ops, det
    if nl1 and not nl2:
        ops.add("translit_to_latin")
        return ops, det
    if ACCENT.search(n2) and not ACCENT.search(n1):
        ops.add("accent_add")
    if (WEB.search(n2) and not WEB.search(n1)) or (bare(n2).endswith("com") and not bare(n1).endswith("com")):
        ops.add("web_form")
        core1 = bare(name_core(n1)); b2 = bare(re.sub(r"(?i)www\.|\.com|@", " ", n2))
        if core1 and (core1 in b2 or core1 in bare(n2)):
            return ops, det
    if bare(n1) == bare(n2):
        ops.add("punct_space_only")
        return ops, det
    p1, p2 = punct(n1), punct(n2)
    if p1 != p2:
        ops.add("punct_change")
        for ch in (p2 - p1):
            det.append(("punct_add", ch))
        for ch in (p1 - p2):
            det.append(("punct_del", ch))
    if ("(" in n2 or "[" in n2) and not ("(" in n1 or "[" in n1):
        ops.add("bracket_add")
    if PH_RE.search(n2) and not PH_RE.search(n1):
        ops.add("phrase_add")
        det.append(("phrase", PH_RE.search(n2).group(0).lower()))
    tsr = fuzz.token_set_ratio(name_core(n1), name_core(n2))
    if tsr < 40:
        ops.add("brand_replace")
        return ops, det
    t1, t2 = ntoks(n1), ntoks(n2)
    L1 = collections.Counter(x for x in t1 if x in LEG); L2 = collections.Counter(y for y in t2 if y in LEG)
    if L1 != L2:
        k = "legal_drop" if not L2 else "legal_add" if not L1 else "legal_drop_part" if not (L2 - L1) else             "legal_add_part" if not (L1 - L2) else "legal_swap"
        ops.add(k); det.append((k, " ".join(sorted(L1.elements())) + ">" + " ".join(sorted(L2.elements()))))
    c1 = [x for x in t1 if x not in LEG]; c2 = [y for y in t2 if y not in LEG]
    if L1 == L2 and L1 and c1 == c2 and t1 != t2:
        ops.add("legal_reorder")
    if c1 != c2 and sorted(c1) == sorted(c2):
        ops.add("reorder")
        c1 = c2
    sm = difflib.SequenceMatcher(None, c1, c2, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = c1[i1:i2], c2[j1:j2]
        if tag == "equal":
            continue
        if tag == "delete":
            for x in a:
                ops.add("word_drop"); det.append(("word_drop", x))
        elif tag == "insert":
            for y in b:
                k = "honorific_add" if y in HON else "phrase_word" if y in PHRASE else "dup_word" if y in c1 else "word_add"
                ops.add(k); det.append((k, y))
        else:
            if "".join(a) == "".join(b):
                ops.add("space_split_join"); continue
            if len(a) == len(b):
                for x, y in zip(a, b):
                    k = tok_pair(x, y); ops.add(k); det.append((k, f"{x}>{y}"))
            else:
                ops.add("multi_replace"); det.append(("multi_replace", " ".join(a) + ">" + " ".join(b)))
    if not ops - {k for k in ops if k.startswith("case:")}:
        ops.add("other")
    return ops, det


UNITW = {"unit", "apt", "ste", "suite", "fl", "floor", "bldg", "room", "rm", "no", "flat", "plot", "shop", "door", "h",
         "sec", "sector", "block", "pmb", "box", "po", "lot", "spc", "space", "trlr", "dept", "hno"}


def first_num(t):
    for i, x in enumerate(t):
        if x[:1].isdigit() and (i == 0 or t[i - 1] not in UNITW):
            return x
    return ""


def postal_of(t):
    return next((x for x in reversed(t) if x.isdigit() and len(x) in (5, 6)), "")


def house_kind(h1, h2):
    if h1 == h2:
        return "same" if h1 else "none"
    if not h2:
        return "dropped"
    if not h1:
        return "added"
    if not (h1.isdigit() and h2.isdigit()):
        return "alnum_change"
    if len(h1) == len(h2) and sorted(h1) == sorted(h2):
        return "digit_swap"
    d = LV.distance(h1, h2)
    if d == 1:
        return "one_digit_" + ("sub" if len(h1) == len(h2) else "del" if len(h2) < len(h1) else "ins")
    diff = abs(int(h1) - int(h2))
    return "near<=10" if diff <= 10 else "near<=100" if diff <= 100 else "random"


def addr_ops(a1, a2):
    e1, e2 = not (a1 or "").strip(), not (a2 or "").strip()
    if e1 and e2:
        return {"both_empty"}, "none", "none", []
    if e2:
        return {"copy_empty"}, "n/a", "n/a", []
    if e1:
        return {"s1_empty"}, "n/a", "n/a", []
    if a1 == a2:
        return {"exact"}, "same", "same", []
    ops, det = set(), []
    c1, c2 = casepat(a1), casepat(a2)
    if c1 != c2:
        ops.add(f"case:{c1}>{c2}")
    if a1.lower() == a2.lower():
        ops.add("case_only"); return ops, "same", "same", det
    if bare(a1) == bare(a2):
        ops.add("punct_space_only"); return ops, "same", "same", det
    comps1 = [c.strip().lower() for c in a1.split(",") if c.strip()]
    comps2 = [c.strip().lower() for c in a2.split(",") if c.strip()]
    if comps1 != comps2 and sorted(comps1) == sorted(comps2):
        ops.add("component_reorder")
    t1, t2 = addr_tokens(a1), addr_tokens(a2)
    h1, h2 = first_num(t1), first_num(t2)
    p1, p2 = postal_of(t1), postal_of(t2)
    hk = house_kind(h1, h2)
    pk = "same" if p1 == p2 else "dropped" if not p2 else "added" if not p1 else \
        "one_digit" if LV.distance(p1, p2) == 1 else "changed"
    if hk not in ("same", "none"):
        ops.add("house_" + hk)
    if pk != "same":
        ops.add("postal_" + pk)
    # raw-token abbreviation changes
    r1 = collections.Counter(TOK.findall(unidecode(a1).lower())); r2 = collections.Counter(TOK.findall(unidecode(a2).lower()))
    rm, ad = r1 - r2, r2 - r1
    nad = {ADDR_ABBR.get(y, y) for y in ad}
    if any(ADDR_ABBR.get(x, x) in nad and ADDR_ABBR.get(x, x) != x or (ADDR_ABBR.get(x, x) in ad) for x in rm) or \
       any(ADDR_ABBR.get(y, y) in {ADDR_ABBR.get(x, x) for x in rm} for y in ad if y not in rm):
        ops.add("abbrev_change")
    # remaining word-level changes after normalisation (house/postal removed)
    w1 = [x for x in t1 if x not in (h1, p1)]; w2 = [x for x in t2 if x not in (h2, p2)]
    if w1 != w2:
        if sorted(w1) == sorted(w2):
            ops.add("token_reorder")
        else:
            sm = difflib.SequenceMatcher(None, w1, w2, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                a, b = w1[i1:i2], w2[j1:j2]
                if tag == "delete":
                    ops.add("tok_drop"); det += [("tok_drop", x) for x in a]
                elif tag == "insert":
                    ops.add("tok_add"); det += [("tok_add", y) for y in b]
                elif tag == "replace":
                    if "".join(a) == "".join(b):
                        ops.add("space_split_join"); continue
                    if len(a) == len(b):
                        for x, y in zip(a, b):
                            k = "tok_typo" if (LV.distance(x, y) <= 1 and max(len(x), len(y)) >= 3) or \
                                (LV.distance(x, y) == 2 and max(len(x), len(y)) >= 6) else \
                                "tok_abbrev" if (x.startswith(y) or y.startswith(x)) else "tok_replace"
                            ops.add(k); det.append((k, f"{x}>{y}"))
                    else:
                        ops.add("tok_multi_replace"); det.append(("tok_multi_replace", " ".join(a) + ">" + " ".join(b)))
    if not ops - {k for k in ops if k.startswith("case:")}:
        ops.add("abbrev_or_norm_only")
    return ops, hk, pk, det


rows = []
ndet = collections.defaultdict(collections.Counter)
adet = collections.defaultdict(collections.Counter)
for s1r, mr, src, ctry, n1, a1, n2, a2 in tp.select("s1r", "mr", "msrc", "country", "n1", "a1", "n2", "a2").iter_rows():
    no, nd = name_ops(n1 or "", n2 or "")
    ao, hk, pk, ad = addr_ops(a1 or "", a2 or "")
    key = (ctry, src)
    for k, v in nd:
        ndet[(ctry, k)][v] += 1
    for k, v in ad:
        adet[(ctry, k)][v] += 1
    tsr = fuzz.token_set_ratio(name_core(n1 or ""), name_core(n2 or ""))
    rows.append((s1r, mr, src, ctry, "|".join(sorted(no)), "|".join(sorted(ao)), hk, pk, tsr))
df = pl.DataFrame(rows, schema=["s1r", "mr", "src", "country", "nops", "aops", "house_k", "postal_k", "name_tsr"], orient="row")
df.write_parquet(f"/work/dive/g1v2_pairs_{PART}.parquet")
import pickle
pickle.dump((dict(ndet), dict(adet)), open(f"/work/dive/g1v2_det_{PART}.pkl", "wb"))
print("done", PART, df.height)
