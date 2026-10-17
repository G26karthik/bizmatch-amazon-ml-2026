"""Deep dive 25: why are strict-address single-S1 records with near-identical names unassigned in test? Cap status of
the S1 (per source), production probability if in the uncertain zone, candidate status, and raw examples.

uv run --no-project --with polars==1.44.2 python scripts/deep/d25_strict_check.py
"""
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
SP = "local_data/work"
te = pl.read_parquet(f"{SP}/strict_addr_test.parquet")
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
nm = dict(zip(rec["entity_id"], rec["business_name"])); ad = dict(zip(rec["entity_id"], rec["business_address"])); rid = dict(zip(rec["entity_id"], rec["rid"]))
g = pl.read_csv("output/v28q_fr20/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
mem = {s: [e for e in (m or "").split(",") if e] for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].to_list())}
unc = pl.read_parquet("local_data/llm/unc_test.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
best = {q: (s, p) for q, s, p in unc.select("q", "s1", "prob").iter_rows()}
cp = pl.read_csv("output/v28q_fr20/candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = defaultdict(set)
for s, m in zip(cp["source1_entity_id"].to_list(), cp["candidate_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        cand[q].add(s)
rows = []
for c, m, s, sim in te.select("country", "m", "s1", "sim").iter_rows():
    src = m[:2]; n_src = sum(1 for x in mem.get(s, []) if x[:2] == src)
    full = n_src >= (5 if src == "S2" else 6)
    b = best.get(rid[m])
    zone = "none" if b is None else ("best=S1" if b[0] == rid[s] else "best=other")
    rows.append((c, m, s, sim, full, n_src, zone, b[1] if b else None, s in cand.get(m, ())))
x = pl.DataFrame(rows, schema=["country", "m", "s1", "sim", "full", "n_src", "zone", "p", "is_cand"], orient="row")
x = x.with_columns(pl.col("sim").cut([80, 90]).alias("simb"))
print(x.group_by("country", "simb", "full", "is_cand", "zone").agg(pl.len(), pl.col("p").mean().round(3)).sort("country", "simb", "full", "is_cand", "zone"))
for c in ("India", "US"):
    ex = x.filter((pl.col("country") == c) & (pl.col("sim") > 90) & ~pl.col("full")).sample(12, seed=5)
    print(c, "examples (sim>90, room under cap):")
    for m, s, zone, p, ic in ex.select("m", "s1", "zone", "p", "is_cand").iter_rows():
        print(f"   {nm[m]!r:45.45s} | {ad[m]!s:45.45s} -> {nm[s]!r:40.40s} | {ad[s]!s:40.40s} [{zone} p={p} cand={ic}]")
x.write_parquet(f"{SP}/strict_addr_test_status.parquet")
