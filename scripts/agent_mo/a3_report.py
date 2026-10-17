"""agent_mo task A report: per-class train precision, test counts by fr27 status, decoy-excess / implied precision.

uv run --no-project --with polars==1.44.2 python scripts/agent_mo/a3_report.py [samples]
"""
import os
import sys

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45)
N1 = pl.DataFrame({"country": ["US", "India"], "S1tr": [1323633, 883188], "S1te": [663106, 809986]})
cs = [c for c in ("US", "India") if os.path.exists(f"local_data/work/agent_mo_A_train_{c}.parquet") and os.path.exists(f"local_data/work/agent_mo_A_test_{c}.parquet")]
tr = pl.concat([pl.read_parquet(f"local_data/work/agent_mo_A_train_{c}.parquet") for c in cs], how="diagonal_relaxed")
te = pl.concat([pl.read_parquet(f"local_data/work/agent_mo_A_test_{c}.parquet") for c in cs], how="diagonal_relaxed")
tr = tr.with_columns(pl.col("ncore").clip(1, 3).alias("ncorec"), pl.col("ntok").clip(1, 5).alias("ntokc"))
te = te.with_columns(pl.col("ncore").clip(1, 3).alias("ncorec"), pl.col("ntok").clip(1, 5).alias("ntokc"))


def table(gb):
    a = tr.group_by(gb).agg(pl.len().alias("tr_n"), pl.col("right").sum().alias("tr_right"))
    b = te.group_by(gb).agg(pl.len().alias("te_n"), (pl.col("status") == "acc_same").sum().alias("te_acc"),
                            (pl.col("status") == "acc_other").sum().alias("te_oth"), (pl.col("status") == "unassigned").sum().alias("te_un"))
    x = a.join(b, on=gb, how="full", coalesce=True).fill_null(0).join(N1, on="country")
    x = x.with_columns((pl.col("tr_right") / pl.col("tr_n")).round(4).alias("P_train"),
                       (pl.col("tr_n") / pl.col("S1tr") * 1000).round(3).alias("tr_all_1k"),
                       (pl.col("tr_right") / pl.col("S1tr") * 1000).round(3).alias("tr_copy_1k"),
                       (pl.col("te_n") / pl.col("S1te") * 1000).round(3).alias("te_all_1k"),
                       (pl.col("te_acc") / pl.col("S1te") * 1000).round(3).alias("te_acc_1k"),
                       (pl.col("te_un") / pl.col("S1te") * 1000).round(3).alias("te_un_1k"))
    x = x.with_columns((pl.col("te_all_1k") / (pl.col("tr_all_1k") + 1e-9)).round(3).alias("all_ratio"),
                       ((pl.col("tr_copy_1k") - pl.col("te_acc_1k")) / (pl.col("te_un_1k") + 1e-9)).round(3).alias("P_impl_un"))
    return x.drop("S1tr", "S1te").sort(gb)


for gb in (["country"], ["country", "op"], ["country", "op", "ncorec"]):
    print(table(gb))
if len(sys.argv) > 1:
    rec = {}
    for split in ("train", "test"):
        rec[split] = pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet").select("entity_id", "business_name") for s in (1, 2, 3)])
    for c in cs:
        for nm, d, split in (("TRAIN wrong", tr.filter((pl.col("country") == c) & ~pl.col("right")), "train"), ("TRAIN right", tr.filter((pl.col("country") == c) & pl.col("right")), "train"),
                             ("TEST unassigned", te.filter((pl.col("country") == c) & (pl.col("status") == "unassigned")), "test")):
            d = d.sample(min(12, d.height), seed=1)
            ids = pl.concat([d["m"], d["s1"]]).implode()
            nmap = rec[split].filter(pl.col("entity_id").is_in(ids)).collect()
            nmap = dict(zip(nmap["entity_id"], nmap["business_name"]))
            print(c, nm)
            for r in d.iter_rows(named=True):
                print("   ", r["op"], "|", nmap.get(r["m"]), "||", nmap.get(r["s1"]))
