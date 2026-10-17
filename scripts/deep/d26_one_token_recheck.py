"""Deep dive 26: re-check the fr20 one-token adds: exact equality of ALL digit tokens of the two addresses, model status
(uncertain-zone best pair / probability) and examples per country x kind.

uv run --no-project --with polars==1.44.2 python scripts/deep/d26_one_token_recheck.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
SP = "local_data/work"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); rid = dict(zip(rec["entity_id"], rec["rid"]))
t = pl.read_parquet(f"{SP}/one_token_rule_test.parquet").filter(pl.col("kind").is_in(["initials", "handle"]) | ((pl.col("kind") == "pseudo") & (pl.col("n_cand") == 1)))
g = pl.read_csv("output/v28q_fr19/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
unc = pl.read_parquet("local_data/llm/unc_test.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
best = {q: (s, p) for q, s, p in unc.select("q", "s1", "prob").iter_rows()}
rows = []
for c, m, s, k in t.select("country", "m", "s1", "kind").iter_rows():
    if m in assigned:
        continue
    dm = sorted(x for x in toks(ad[m]) if x.isdigit()); ds = sorted(x for x in toks(ad[s]) if x.isdigit())
    dm2 = [x.lstrip("0") for x in dm]; ds2 = [x.lstrip("0") for x in ds]
    b = best.get(rid[m])
    rows.append((c, k, m, s, dm2 == ds2, "none" if b is None else ("best=S1" if b[0] == rid[s] else "best=other"), b[1] if b else None))
x = pl.DataFrame(rows, schema=["country", "kind", "m", "s1", "digits_eq", "zone", "p"], orient="row")
print(x.group_by("country", "kind", "digits_eq", "zone").agg(pl.len(), pl.col("p").mean().round(3)).sort("country", "kind", "digits_eq", "zone"))
for c, k in (("India", "initials"), ("India", "pseudo"), ("US", "pseudo"), ("France", "pseudo")):
    ex = x.filter((pl.col("country") == c) & (pl.col("kind") == k)).sample(8, seed=1)
    print(c, k)
    for m, s, de, z, p in ex.select("m", "s1", "digits_eq", "zone", "p").iter_rows():
        print(f"   {nm[m]!r:14.14s} | {ad[m]!s:50.50s} -> {nm[s]!r:36.36s} | {ad[s]!s:45.45s} [{de} {z} {p}]")
x.write_parquet(f"{SP}/one_token_recheck.parquet")
