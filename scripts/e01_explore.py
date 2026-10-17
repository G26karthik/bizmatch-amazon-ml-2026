"""E01 exploration: raw French address tokens around the normaliser gaps (bis/ter/cours/quai/passage/residence/app)."""
import re
from collections import Counter

import polars as pl
from unidecode import unidecode

r = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_address", "addr_n", "house"])
f = r.with_row_index("rid").filter(pl.col("country") == "France")
pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_rows(60)
print(f.select("src", "business_address", "addr_n", "house").sample(25, seed=1))
TOK = re.compile(r"[a-z0-9]+|[^a-z0-9\s]")
KEYS = {"bis", "b", "ter", "t", "cours", "crs", "quai", "q", "passage", "psg", "pass", "residence", "res", "appartement",
        "app", "apt", "appt", "bat", "batiment", "bt", "a", "c", "d", "n", "no", "ndeg", "deg", "et", "etage", "esc", "lot", "zi", "za", "zac", "bp", "cedex", "cs"}
for s in (1, 2, 3):
    c = Counter()
    for a in f.filter(pl.col("src") == s)["business_address"].to_list():
        for t in re.findall(r"[a-z0-9]+", unidecode(a or "").lower()):
            if t in KEYS or re.fullmatch(r"\d+[a-z]+", t):
                c[t if t in KEYS else "NUM+" + re.sub(r"\d", "", t)] += 1
    print(s, c.most_common(60))
# raw examples with a key token
for key in ["bis", r"\d+b\b", r"\d+ b\b", "ter", r"\bcours\b", r"\bcrs\b", r"\bquai\b", r"\bq\b", r"\bpassage\b", r"\bpsg\b", r"\bpass\b",
            r"r[eé]sidence", r"\br[eé]s\b", "appartement", r"\bapp\b", r"\bapt\b", r"\bappt\b", r"\bbat\b", r"\bt\b"]:
    x = f.filter(pl.col("business_address").str.to_lowercase().str.contains(key))
    print(f"\n== {key}: S1 {x.filter(pl.col('src')==1).height} S2 {x.filter(pl.col('src')==2).height} S3 {x.filter(pl.col('src')==3).height}")
    print(x.select("src", "business_address", "addr_n", "house").sample(min(6, x.height), seed=2))
