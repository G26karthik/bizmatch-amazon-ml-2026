"""Accepted France pairs whose record name is a web handle ("anantacompagniesas.com", "#poleunion", "@x"): compare the
handle core with concatenations of the S1 name tokens. Classes: exact (all tokens / without legal / without stop words),
prefix-initials forms (e.g. "tfcomite" = initials + last word), one S1 word replaced by another French S1 vocabulary word
(handle of a sibling decoy name -> two operations -> decoy), other.

python scripts/fr_handle_audit.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
HANDLE = re.compile(r"^\s*[#@]?([a-z0-9&+\-]+)(\.(?:com|fr|net|org|eu|io|co))?\s*$")


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name", "src", "country"])
nm = rec["business_name"].to_list()
voc = [w for w, c in Counter(w for n in rec.filter((pl.col("country") == "France") & (pl.col("src") == 1))["business_name"].to_list()
                             for w in set(re.findall(r"[a-z0-9]+", norm(n)))).most_common(4000) if c > 30 and len(w) >= 3]
x = pl.read_parquet("/work/lead/fr_name_ops_audit.parquet")
rows = []
for q, s, pat in zip(x["q"].to_list(), x["s1"].to_list(), x["pat"].to_list()):
    r = norm(nm[q])
    m = HANDLE.match(r)
    if not m or not (m.group(2) or r.strip().startswith(("#", "@"))):
        continue
    h = re.sub(r"[^a-z0-9]", "", m.group(1))
    toks = re.findall(r"[a-z0-9]+", norm(nm[s]))
    core = [t for t in toks if t not in LEGF]
    core2 = [t for t in core if t not in STOP]
    cands = {"".join(toks), "".join(core), "".join(core2)}
    if h in cands:
        cls = "exact"
    elif any(h == "".join(t[0] for t in core2[:i]) + "".join(core2[i:]) for i in range(1, len(core2))):
        cls = "initials_prefix"
    elif any(c.startswith(h) or h.startswith(c) for c in cands if c):
        cls = "prefix"
    else:
        cls = "other"
        for i, t in enumerate(core2):  # one word replaced by another vocabulary word
            pre, post = "".join(core2[:i]), "".join(core2[i + 1:])
            if h.startswith(pre) and h.endswith(post) and len(h) > len(pre) + len(post):
                mid = h[len(pre):len(h) - len(post)] if post else h[len(pre):]
                if mid != t and mid in voc:
                    cls = "vocab_swap"
                    break
                if mid != t:
                    cls = "other_swap"
    rows.append((q, s, pat, cls, h))
y = pl.DataFrame(rows, schema=["q", "s1", "pat", "hcls", "handle"], orient="row")
print("accepted France web-handle pairs:", y.height)
print(y.group_by("hcls").len().sort("len", descending=True))
nm1 = {s: nm[s] for s in y["s1"].to_list()}
for c in ("vocab_swap", "other_swap", "other"):
    z = y.filter(pl.col("hcls") == c)
    print(f"\nexamples {c} ({z.height})")
    for q, s, h in z.sample(n=min(12, z.height), seed=10).select("q", "s1", "handle").iter_rows():
        print(f"   {nm[s]!r:42} | {nm[q]!r}")
y.write_parquet("/work/lead/fr_handle_audit.parquet")
