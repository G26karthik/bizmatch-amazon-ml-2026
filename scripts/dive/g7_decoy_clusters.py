"""Generator dive 7 (worker): do decoys come in clusters (a fake entity with its own S2+S3 copies)?
Within each sampled S1 group (groups_sample.parquet from g4): decoy-decoy pairs vs copy-copy pairs.
"""
import itertools

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
g = pl.read_parquet("/work/dive/groups_sample.parquet")
rows = []
for (s1r,), grp in g.group_by(["s1r"]):
    S = grp.filter(pl.col("kind") == "s").to_dicts()
    if not S:
        continue
    S = S[0]
    st = set((S["name_n"] or "").split())
    D = grp.filter(pl.col("kind") == "d").to_dicts()
    for d in D:
        d["extra"] = frozenset(set((d["name_n"] or "").split()) - st)
    for a, b in itertools.combinations(D, 2):
        rows.append(dict(country=S["country"], pair="".join(sorted([f"S{a['src']}", f"S{b['src']}"])),
                         strong=(a["pA"] or 0) >= 0.5 and (b["pA"] or 0) >= 0.5,
                         name_eq=(a["name_n"] or "") == (b["name_n"] or ""), extra_eq=a["extra"] == b["extra"] and len(a["extra"]) > 0,
                         extra_overlap=len(a["extra"] & b["extra"]) > 0,
                         house_eq=(a["house"] or "") == (b["house"] or "") and bool(a["house"]),
                         both_addr=bool(a["addr_n"]) and bool(b["addr_n"]),
                         addr_ratio=fuzz.ratio(a["addr_n"] or "", b["addr_n"] or ""),
                         name_ratio=fuzz.ratio(a["name_n"] or "", b["name_n"] or "")))
r = pl.DataFrame(rows)
print("decoy-decoy pairs within one S1 group:", r.height)
print(r.group_by("country", "strong", "pair").agg(
    pl.len(), pl.col("name_eq").mean().round(3), pl.col("extra_eq").mean().round(3), pl.col("extra_overlap").mean().round(3),
    pl.col("house_eq").filter(pl.col("both_addr")).mean().round(3).alias("house_eq|addr"),
    (pl.col("addr_ratio") >= 90).filter(pl.col("both_addr")).mean().round(3).alias("addr>=90|addr"),
    pl.col("name_ratio").mean().round(1)).sort("country", "strong", "pair"))
# per S1 group: number of decoys by source, and whether an S2 decoy has an S3 twin (same extra words)
d = g.filter(pl.col("kind") == "d")
per = d.group_by("s1r").agg((pl.col("src") == 2).sum().alias("d2"), (pl.col("src") == 3).sum().alias("d3"),
                            (pl.col("pA") >= 0.5).sum().alias("dstrong"))
print("decoys per S1 (d2 x d3, S1 groups with >=1 decoy):")
print(per.group_by("d2", "d3").len().sort("len", descending=True).head(15))
print("strong decoys per S1:", per.group_by("dstrong").len().sort("dstrong").to_dicts())
