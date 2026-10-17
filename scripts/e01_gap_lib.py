"""E01 shared helpers (name relation, tokens, gap canonicalisation), copied from scripts/e01_gap.py."""
import re
import unicodedata
from collections import Counter

from rapidfuzz.distance import Levenshtein

WORD = {"crs": "cours", "psg": "passage", "pass": "passage", "res": "residence", "app": "appartement",
        "apt": "appartement", "appt": "appartement", "q": "quai"}
AFTER_NUM = {"b": "bis", "t": "ter"}
# street-type tokens (after the base normaliser): "12 q r ..." is 12 quater rue, "57 q de bacalan" is 57 quai de bacalan
STREET = {"r", "rue", "ave", "blvd", "ch", "all", "imp", "rte", "pl", "cours", "crs", "quai", "passage", "psg", "pass", "sq",
          "cite", "cour", "residence", "res", "faubourg", "sente", "sentier", "voie", "villa", "hameau", "chaussee", "parvis",
          "prom", "promenade", "esplanade", "mail", "rond", "lot", "lotissement", "clos", "domaine", "montee", "traverse",
          "square", "quartier", "zi", "za", "zac", "venelle", "ruelle", "carrefour", "galerie"}
FULL = {"bis", "ter", "quater", "cours", "quai", "passage", "residence", "appartement"}
SUF = {"bis", "ter", "quater"}
GAPCLASS = {"bis": "bis", "b": "bis", "ter": "ter", "t": "ter", "quater": "quater", "cours": "cours", "crs": "cours",
            "quai": "quai", "q": "quai", "passage": "passage", "psg": "passage", "pass": "passage", "residence": "residence",
            "res": "residence", "appartement": "appartement", "app": "appartement", "apt": "appartement", "appt": "appartement"}


def canon(tokens):
    """gap abbreviations -> one canonical form; b/t/q are house suffixes only right after a number (q only before a
    street type: '12 q r x' = 12 quater, '57 q de x' = quai). House number, suffix and street stay separate tokens."""
    out = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        after_num = i > 0 and tokens[i - 1][:1].isdigit()
        nxt = tokens[i + 1] if i + 1 < n else ""
        if after_num and t in AFTER_NUM:
            out.append(AFTER_NUM[t])
        elif after_num and t == "q" and nxt in STREET:
            out.append("quater")
        else:
            out.append(WORD.get(t, t))
    return out


def house_suffix(tokens):
    """canonical suffix right after the first number ('' if none)."""
    c = canon(tokens)
    for i, t in enumerate(tokens):
        if t[:1].isdigit():
            return c[i + 1] if i + 1 < len(c) and c[i + 1] in SUF else ""
    return ""


def gap_types(ta, tb):
    """gap classes (canonical forms) of the tokens that differ before canonicalisation."""
    ca_, cb_ = Counter(ta), Counter(tb)
    diff = set(((ca_ - cb_) + (cb_ - ca_)).elements())
    cls = set()
    for toks_ in (ta, tb):
        for t, c in zip(toks_, canon(toks_)):
            if t in diff and (t != c or t in FULL):
                cls.add(c)
    return ",".join(sorted(cls)) or "?"


STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s",
        "a", "u", "r", "e"}
LEGALW = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scop", "sca", "scs", "gie"}
DECOY = {"holding", "participations", "international", "distribution", "groupe", "developpement", "france"}


_ACRO = re.compile(r"(?<![a-z])(?:[a-z]\.){2,}[a-z]?\.?")


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = _ACRO.sub(lambda m: m.group(0).replace(".", ""), s)  # "s.c.i." -> "sci"
    return re.findall(r"[a-z0-9]+", s)


def is_initials(nq, ns):
    """record name is the initials of the S1 name ("AB" for "Alpha Beta SA"), a copy name operation."""
    tq = [t for t in toks(nq) if t not in LEGALW]
    ts = [t for t in toks(ns) if t not in LEGALW]
    if len(tq) != 1 or len(tq[0]) < 2 or len(ts) < 2:
        return False
    return tq[0] in ("".join(t[0] for t in ts), "".join(t[0] for t in ts if t not in STOP))


def name_rel(nq, ns):
    """exact | case | legal (only legal-form words differ) | typo (one token within edit 2) | drop1 | add1 | swap1 | other"""
    if nq == ns:
        return "exact"
    if nq.lower() == ns.lower():
        return "case"
    tq, ts = toks(nq), toks(ns)
    if tq == ts:
        return "punct"
    bq, bs = [t for t in tq if t not in LEGALW], [t for t in ts if t not in LEGALW]
    if bq == bs:
        return "legal"
    sq, ss = Counter(bq), Counter(bs)
    onlyq, onlys = list((sq - ss).elements()), list((ss - sq).elements())
    if len(onlyq) == 1 and len(onlys) == 1:
        if Levenshtein.distance(onlyq[0], onlys[0]) <= 2 and min(len(onlyq[0]), len(onlys[0])) >= 3:
            return "typo"
        return "swap1:" + onlys[0] + ">" + onlyq[0]
    if not onlyq and len(onlys) == 1:
        return "drop1:" + onlys[0]
    if not onlys and len(onlyq) == 1:
        return "add1:" + onlyq[0]
    if not onlyq and not onlys:
        return "reorder"
    return "other"


