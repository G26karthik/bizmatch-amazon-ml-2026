"""Text normalization for business names and addresses (country-agnostic; no external data)."""
from __future__ import annotations

import re

from unidecode import unidecode

_TOKEN = re.compile(r"[a-z0-9]+")

ADDR_ABBR = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "court": "ct", "place": "pl", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "square": "sq", "trail": "trl", "mount": "mt", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "suite": "ste",
    "apartment": "apt", "floor": "fl", "building": "bldg", "number": "no", "nagar": "ngr", "marg": "mrg",
    "sector": "sec", "near": "nr", "opposite": "opp", "opp": "opp", "first": "1st", "second": "2nd",
    "third": "3rd", "fourth": "4th", "fifth": "5th", "ground": "grd", "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private",
    "plc", "lp", "llp", "pc", "pllc", "the", "and", "of", "sa", "sas", "sarl", "eurl", "sasu", "sci", "snc",
    "gmbh", "opc", "pvtltd", "dba",
}
ADDR_ABBR.update({"bd": "blvd", "bvd": "blvd", "fbg": "faubourg", "pl": "pl", "imp": "imp", "che": "ch",
                  "avenida": "ave", "gali": "gali", "mohalla": "mohalla"})
ORDINAL = {"st", "nd", "rd", "th", "er", "e", "eme"}
_NUM_SUFFIX = re.compile(r"^(\d+)([a-z]+)$")
NULLISH = {"null", "none", "nan", "na"}
_NA_RE = re.compile(r"\bn\s*/\s*a\b")


def ascii_lower(s: str) -> str:
    return unidecode(s or "").lower().replace("&", " and ")


def name_tokens(s: str) -> list[str]:
    return _TOKEN.findall(ascii_lower(s))


def name_core(s: str) -> str:
    """Name without legal suffixes / stopwords; used for similarity features."""
    toks = [t for t in name_tokens(s) if t not in LEGAL]
    return " ".join(toks) if toks else " ".join(name_tokens(s))


_DIGRAPH = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
            ("dh", "d"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("c", "k"), ("w", "b"), ("v", "b"),
            ("z", "j")]
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant skeleton robust to transliteration / vowel noise: 'laaiph' -> 'lf', 'life' -> 'lf'."""
    for a, b in _DIGRAPH:
        tok = tok.replace(a, b)
    head, tail = tok[:1], _VOWELS.sub("", tok[1:])
    return _REPEAT.sub(r"\1", head + tail)


_LEGAL_SKEL = {skeleton(w) for w in LEGAL if len(w) > 3}


def name_skel(s: str) -> str:
    """Skeleton tokens of the name, dropping legal words even when transliterated ('praaivett' -> 'prvt')."""
    out = []
    for t in name_tokens(s):
        if t in LEGAL:
            continue
        k = skeleton(t)
        if len(k) > 1 and k not in _LEGAL_SKEL:
            out.append(k)
    return " ".join(out)


def _num(t: str) -> str:
    return t.lstrip("0") or "0" if t.isdigit() else t


def addr_tokens(s: str) -> list[str]:
    s = ascii_lower(s)
    s = _NA_RE.sub(" ", s)
    s = _STATE_RE.sub(lambda m: US_STATES[m.group(1)], s)
    out = []
    for t in _TOKEN.findall(s):
        if t in NULLISH:
            continue
        m = _NUM_SUFFIX.match(t)
        if m and m.group(2) not in ORDINAL:  # "45bis" -> 45 bis, "624c" -> 624 c; keep "1st"
            out.append(_num(m.group(1)))
            out.append(m.group(2))
            continue
        out.append(_num(ADDR_ABBR.get(t, t)))
    return out


def addr_norm(s: str) -> str:
    return " ".join(addr_tokens(s))


def numbers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t[:1].isdigit()]
