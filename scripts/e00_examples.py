"""E00 part 4: compact examples per bucket from /work/agent_e00/err_records.parquet (one line per record)."""
import sys
from pathlib import Path

import polars as pl

OUT = Path("/work/agent_e00")
NB = 89043
e = pl.read_parquet(OUT / "err_records.parquet")
rec = pl.read_parquet("/work/run28/train/records.parquet", columns=["business_name", "business_address"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
e = e.join(rec.select(pl.col("rid").alias("o_c"), pl.col("business_name").alias("o_raw_name"), pl.col("business_address").alias("o_raw_addr")),
           on="o_c", how="left")
rr = OUT / "retr_ranks.parquet"
if rr.exists():
    e = e.join(pl.read_parquet(rr).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "rank_c", "rank_a", "rank_n", "in_e5"),
               on=["q", "s1"], how="left")


def show(df, title, n=12):
    print(f"\n##### {title}: {df.height} records, {df['loss'].sum() / NB:.6f} F0.5")
    for r in df.sort("loss", descending=True).head(n).iter_rows(named=True):
        extra = f" rank_n={r.get('rank_n')} rank_c={r.get('rank_c')}" if r.get("rank_n") is not None else ""
        print(f"[{r['cat']}|{r['country']}|S{r['q_src']}|T={r['T']}|p={r['am_p'] if r['am_p'] is None else round(r['am_p'], 3)}|pA={r['pA'] if r['pA'] is None else round(r['pA'], 4)}"
              f"|psc={r['p_sc'] if r['p_sc'] is None else round(r['p_sc'], 1)} osc={r['o_sc'] if r['o_sc'] is None else round(r['o_sc'], 1)}"
              f"|nameS1cnt={r['s_name_cnt']} qnamecnt={r['q_name_cnt']} acc={r.get('acc') if r.get('acc') is None else round(r['acc'], 2)}{extra}]")
        print(f"    rec: {r['q_raw_name']!r} @ {r['q_raw_addr']!r}")
        print(f"    S1 : {r['s_raw_name']!r} @ {r['s_raw_addr']!r}")
        if r.get("o_raw_name") is not None and r["cat"].startswith("FN"):
            print(f"    top competitor S1: {r['o_raw_name']!r} @ {r['o_raw_addr']!r}")


fn = e.filter(pl.col("cat").str.starts_with("FN"))
for cat in fn["cat"].unique().sort().to_list():
    for rv in ("rec_parent_best_clear", "irr_competitor_as_good", "irr_empty_addr_namesake"):
        d = fn.filter((pl.col("cat") == cat) & (pl.col("recov") == rv))
        if d.height:
            show(d, f"{cat} / {rv}", 6 if rv != "rec_parent_best_clear" else 10)
show(e.filter(pl.col("cat").str.starts_with("FP")), "FP (all)", 12)
