"""Trim neighbourhood text: keep n/S lines only when same street or clearly similar name."""
import re
import sys
import unicodedata

from rapidfuzz import fuzz

ABBR = {"rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "chemin": "ch", "place": "pl",
        "route": "rte", "impasse": "imp", "allee": "all", "quai": "q", "cours": "crs", "road": "rd", "street": "st",
        "drive": "dr", "lane": "ln", "saint": "st", "no": "", "n": ""}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def aw(s):
    return " ".join(t for t in (ABBR.get(t, t) for t in norm(s).split() if not any(c.isdigit() for c in t)) if t)


def split(l):
    body = l[3:].split("   {")[0] if "   {" in l else l[3:].split("  {")[0]
    body = body.strip()
    body = body.split(" ", 1)[1] if " " in body else ""
    nm, _, ad = body.partition(" | ")
    return nm.strip(), ad.strip()


src, dst = sys.argv[1:3]
out = []
s1n = s1a = ""
for l in open(src, encoding="utf-8"):
    l = l.rstrip("\n")
    if l.startswith(" S1 "):
        nm, _, ad = l[4:].partition(" | ")
        s1n, s1a = norm(nm), aw(ad)
    elif l.startswith(" n ") or l.startswith(" S "):
        nm, ad = split(l)
        n_sort = fuzz.token_sort_ratio(s1n, norm(nm))
        a_set = fuzz.token_set_ratio(s1a, aw(ad)) if ad else 0
        a_sort = fuzz.token_sort_ratio(s1a, aw(ad)) if ad else 0
        same_street = a_set >= 90 and a_sort >= 60
        if not ((same_street and n_sort >= 40) or (n_sort >= 72 and (a_set >= 60 or not ad))):
            continue
    out.append(l)
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(len(out))
