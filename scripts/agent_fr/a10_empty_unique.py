"""Agent FR a10: unassigned French empty-address S2/S3 records whose name key matches exactly one / several French S1s;
model-free cap test at the (first) matching S1 and bracket share; US for calibration (1/3 sample)."""
import re, unicodedata
from collections import Counter
import polars as pl
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
LEG = {"sa","sas","sasu","sarl","eurl","sci","snc","ei","inc","llc","ltd","pvt","private","limited","corp","corporation","incorporated","company","co","lp","llp","pllc","pc","plc","the","de","des","du","la","le","les","et","of","and","s","a","d","l","sarlu","selarl","scp","scm","gie","sca","sel"}
def norm(s): return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
def key(n): return " ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEG))
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id","business_name","business_address","country"]).with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (1,2,3)])
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); ct = rec["country"].to_list(); sr = rec["src"].to_list()
rid = {e:i for i,e in enumerate(eid)}
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = {}; cnt = Counter()
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        assigned[rid[q]] = rid[s]; cnt[(rid[s], sr[rid[q]])] += 1
rows = []
for c in ("France", "US"):
    idx = {}
    for i in range(len(eid)):
        if sr[i] == 1 and ct[i] == c:
            k = key(nm[i])
            if k: idx.setdefault(k, []).append(i)
    for i in range(len(eid)):
        if sr[i] == 1 or ct[i] != c or (ad[i] or "").strip(): continue
        if c == "US" and i % 3: continue
        k = key(nm[i]); cands = idx.get(k)
        if not cands: continue
        st = "acc" if i in assigned else "una"
        s = cands[0] if st == "una" else assigned[i]
        rows.append((c, st, "uniq" if len(cands) == 1 else ("2" if len(cands) == 2 else "3+"), sr[i], cnt[(s, sr[i])] - (1 if st == "acc" else 0), bool(re.search(r"[\[\(\{]", nm[i] or ""))))
x = pl.DataFrame(rows, schema=["country","status","ncand","src","n_other","brk"], orient="row")
full = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
low = pl.col("n_other").is_between(2, 3)
sl = pl.DataFrame([(ct[s], so, cnt[(s, so)]) for s in range(len(eid)) if sr[s] == 1 and ct[s] in ("France", "US") for so in (2, 3)], schema=["country","src","n_other"], orient="row")
for c in ("France", "US"):
    df = sl.filter((pl.col("country") == c) & full).height; dl = sl.filter((pl.col("country") == c) & low).height
    print(c, x.filter(pl.col("country") == c).group_by("status", "ncand").agg(pl.len().alias("n"), pl.col("brk").mean().round(3), pl.col("n_other").mean().round(3).alias("m_other"),
          ((full.sum() / df) / (low.sum() / dl)).round(3).alias("cap")).sort("status", "ncand"))
