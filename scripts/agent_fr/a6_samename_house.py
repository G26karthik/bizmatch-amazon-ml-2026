"""Agent FR a6: S2/S3 records whose name key (tokens minus legal words) equals an S1's name key; best such S1 by street
similarity; status in fr25 (acc to that S1 / other S1 / unassigned) x house relation (same, lev1 digit edit, offset, other,
none) x street sim; cap test for unassigned (rate/slot at full cap vs n_other 2-3). France and US (1/3 sample)."""
import re, sys, unicodedata
from collections import Counter
import polars as pl
from rapidfuzz.distance import Levenshtein
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
SCR = "local_data/work"
LEG = {"sa","sas","sasu","sarl","eurl","sci","snc","ei","inc","llc","ltd","pvt","private","limited","corp","corporation","incorporated","company","co","lp","llp","pllc","pc","plc","the","de","des","du","la","le","les","et","of","and","s","a","d","l","sarlu","selarl","scp","scm","gie","sca","sel"}
OFFS = {1,2,3,4,5,7,9,11,13,21}
def norm(s): return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
def tk(s): return re.findall(r"[a-z0-9]+", norm(s))
src = [pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id","business_name","business_address","country"]).with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (1,2,3)]
rec = pl.concat(src)
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); ct = rec["country"].to_list(); sr = rec["src"].to_list()
rid = {e:i for i,e in enumerate(eid)}
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = {}; cnt = Counter()
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    si = rid[s]
    for q in (m.split(",") if m else []):
        qi = rid[q]; assigned[qi] = si; cnt[(si, sr[qi])] += 1
def addr(i):
    a = tk(ad[i]); h = next((t for t in a if t.isdigit() and len(t) < 7), None)
    return h, set(a) - ({h} if h else set())
rows = []
for c in ("France", "US"):
    idx = {}
    for i in range(len(eid)):
        if sr[i] == 1 and ct[i] == c:
            k = " ".join(sorted(set(tk(nm[i])) - LEG))
            if k: idx.setdefault(k, []).append(i)
    print(c, "index", len(idx), flush=True)
    for i in range(len(eid)):
        if sr[i] == 1 or ct[i] != c or not (ad[i] or "").strip(): continue
        if c == "US" and i % 3: continue
        k = " ".join(sorted(set(tk(nm[i])) - LEG))
        cands = idx.get(k)
        if not cands: continue
        hq, Sq = addr(i)
        best = None
        for s in cands:
            hs, Ss = addr(s)
            j = len(Sq & Ss) / max(1, len(Sq | Ss))
            if best is None or j > best[0]: best = (j, s, hs)
        j, s, hs = best
        if hq is None or hs is None: hr = "nohouse"
        elif hq.lstrip("0") == hs.lstrip("0"): hr = "same_h"
        elif Levenshtein.distance(hq, hs) == 1: hr = "lev1" + ("_off" if abs(int(hq) - int(hs)) in OFFS else "")
        elif abs(int(hq) - int(hs)) in OFFS: hr = "offset"
        else: hr = "h_other"
        a = assigned.get(i)
        st = "acc" if a == s else ("una" if a is None else "other")
        rows.append((c, i, s, st, hr, "j>=.5" if j >= 0.5 else ("j>=.25" if j >= 0.25 else "j<.25"), sr[i], cnt[(s, sr[i])] - (1 if st == "acc" else 0), len(cands), bool(re.search(r"[\[\(\{]", nm[i] or ""))))
x = pl.DataFrame(rows, schema=["country","q","s1","status","hrel","jbin","src","n_other","ncand","brk"], orient="row")
x.write_parquet(f"{SCR}/a6.parquet")
slots = []
for s in range(len(eid)):
    if sr[s] == 1 and ct[s] in ("France", "US"):
        for so in (2, 3): slots.append((ct[s], so, cnt[(s, so)]))
sl = pl.DataFrame(slots, schema=["country","src","n_other"], orient="row")
full = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
low = pl.col("n_other").is_between(2, 3)
for c in ("France", "US"):
    df = sl.filter((pl.col("country") == c) & full).height; dl = sl.filter((pl.col("country") == c) & low).height
    y = x.filter(pl.col("country") == c)
    t = y.group_by("status", "hrel", "jbin").agg(pl.len().alias("n"), (pl.col("src") == 2).mean().round(3).alias("s2"), pl.col("brk").mean().mul(100).round(1).alias("brk"),
        (pl.col("ncand") == 1).mean().round(3).alias("uniq"), ((full.sum() / df) / (low.sum() / dl)).round(3).alias("cap"))
    print(f"=== {c} full slots {df} low {dl}")
    print(t.filter(pl.col("n") >= 100).sort("status", "n", descending=[False, True]))
