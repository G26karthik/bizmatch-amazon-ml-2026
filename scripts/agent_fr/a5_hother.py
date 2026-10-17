"""Agent FR a5: same-name accepted pairs with a different house number (not offset): US vs France examples and
house-digit relation (Levenshtein between house tokens)."""
import re, unicodedata, random
import polars as pl
from rapidfuzz.distance import Levenshtein
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
LEG = {"sa","sas","sasu","sarl","eurl","sci","snc","ei","inc","llc","ltd","pvt","private","limited","corp","corporation","incorporated","company","co","lp","llp","pllc","pc","plc","the","de","des","du","la","le","les","et","of","and","s","a","d","l","sarlu","selarl","scp","scm","gie","sca","sel"}
OFFS = {1,2,3,4,5,7,9,11,13,21}
def norm(s): return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
def tk(s): return re.findall(r"[a-z0-9]+", norm(s))
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id","business_name","business_address","country"]) for s in (1,2,3)])
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); ct = rec["country"].to_list()
rid = {e:i for i,e in enumerate(eid)}
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
random.seed(1); rows=[]; ex={"US":[],"France":[]}
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    si = rid[s]; c = ct[si]
    if c not in ("US","France"): continue
    for q in (m.split(",") if m else []):
        qi = rid[q]
        A = set(tk(nm[qi]))-LEG; B=set(tk(nm[si]))-LEG
        if not A or A!=B: continue
        aq, as_ = tk(ad[qi]), tk(ad[si])
        if not aq: continue
        hq = next((t for t in aq if t.isdigit()), None); hs = next((t for t in as_ if t.isdigit()), None)
        if hq is None or hs is None or len(hq)>6 or len(hs)>6: continue
        d = abs(int(hq)-int(hs))
        if d==0 or d in OFFS: continue
        pos_q = aq.index(hq); pos_s = as_.index(hs)
        hs_in_q = hs in aq
        rows.append((c, Levenshtein.distance(hq,hs), hs_in_q, pos_q==0 and pos_s==0))
        if random.random()<0.02 and len(ex[c])<12: ex[c].append((nm[si], ad[si], nm[qi], ad[qi]))
x = pl.DataFrame(rows, schema=["country","lev","s1house_in_q","both_first"], orient="row")
print(x.group_by("country","lev").len().sort("country","lev"))
print(x.group_by("country").agg(pl.col("s1house_in_q").mean(), pl.col("both_first").mean(), pl.len()))
for c in ex:
    print("==", c)
    for r in ex[c]: print(" | ".join(r))
