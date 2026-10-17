"""Generator dive 9 (worker): explicit rule evaluation on the 30% record sample (allq_{C}.parquet).
Each rule is a condition on (record, its best full-pool S1 candidate X). precision = share of records meeting the
condition that are a true copy of X; coverage = records meeting it (sample count / 0.3 ~ population) and % of S2/S3.
"""
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
US_MARK = ["midtown", "northside", "greater", "riverside", "eastgate", "westgate", "lakeside", "southside"]
rows = []
for C in ("US", "India"):
    b = pl.read_parquet("/work/dive/best2.parquet", columns=["qc", "q", "src"]).filter(pl.col("qc") == C)
    tot = {s: int((b["src"] == s).sum() * 0.3) for s in (2, 3)}  # sample-scale population of S2/S3 records
    a = pl.read_parquet(f"/work/dive/allq_{C}.parquet")
    a = a.with_columns(
        pl.when(pl.col("y")).then(pl.lit("true")).when(pl.col("matched")).then(pl.lit("other")).otherwise(pl.lit("decoy")).alias("cls"),
        pl.col("name_n_2").str.split(" ").list.unique().alias("qt"), pl.col("name_n_1").str.split(" ").list.unique().alias("st"))
    a = a.with_columns(pl.col("qt").list.set_difference("st").alias("extra"), pl.col("st").list.set_difference("qt").alias("miss"))
    hl = cpdist(a["house_1"].fill_null("").to_list(), a["house_2"].fill_null("").to_list(), scorer=Levenshtein.distance, workers=1)
    a = a.with_columns(pl.Series("hlev", np.asarray(hl, dtype=np.int32)),
                       (pl.col("house_1").cast(pl.Int64, strict=False) - pl.col("house_2").cast(pl.Int64, strict=False)).abs().alias("hdiff"))
    nm_eq = pl.col("name_n_1") == pl.col("name_n_2")
    addr_both = ~pl.col("addr_empty_1") & ~pl.col("addr_empty_2")
    h_both = addr_both & (pl.col("house_1") != "") & (pl.col("house_2") != "")
    same_st = pl.col("addr_w_1") == pl.col("addr_w_2")
    pure_add = (pl.col("extra").list.len() > 0) & (pl.col("miss").list.len() == 0)
    has_mark = pl.col("extra").list.eval(pl.element().is_in(US_MARK)).list.any()
    ent = pl.col("extra").list.contains("enterprises")
    swap = pl.col("house_1").str.reverse() == pl.col("house_2")
    R = {
        "name_n equal & house equal": nm_eq & h_both & (pl.col("house_1") == pl.col("house_2")),
        "name_n equal & same street words & house differs": nm_eq & h_both & same_st & (pl.col("house_1") != pl.col("house_2")),
        "  ..house differs by 1 edit (Lev=1)": nm_eq & h_both & same_st & (pl.col("hlev") == 1),
        "  ..house differs by >=2 edits": nm_eq & h_both & same_st & (pl.col("hlev") >= 2),
        "  ..house |diff|<=10": nm_eq & h_both & same_st & (pl.col("hdiff").is_between(1, 10)),
        "name_n equal & house missing in record": nm_eq & addr_both & (pl.col("house_1") != "") & (pl.col("house_2") == ""),
        "name_n equal & record addr empty": nm_eq & pl.col("addr_empty_2"),
        "record addr empty (any name)": pl.col("addr_empty_2"),
        "S1 name + extra word(s) (pure add)": pure_add,
        "  ..pure add & house equal": pure_add & h_both & (pl.col("house_1") == pl.col("house_2")),
        "  ..pure add & same street & house differs": pure_add & h_both & same_st & (pl.col("house_1") != pl.col("house_2")),
        "  ..pure add & record addr empty": pure_add & pl.col("addr_empty_2"),
        "extra word is a US marker": has_mark,
        "  ..marker & house equal": has_mark & h_both & (pl.col("house_1") == pl.col("house_2")),
        "extra word 'enterprises'": ent,
        "  ..enterprises & house equal": ent & h_both & (pl.col("house_1") == pl.col("house_2")),
        "name disjoint (brand replace) & house equal & same street": (pl.col("qt").list.set_intersection("st").list.len() == 0) & h_both & same_st & (pl.col("house_1") == pl.col("house_2")),
        "name disjoint & (house differs or other street)": (pl.col("qt").list.set_intersection("st").list.len() == 0) & addr_both & ~(same_st & (pl.col("house_1") == pl.col("house_2"))),
        "house = digit reversal of S1 house": h_both & swap & (pl.col("house_1") != pl.col("house_2")),
        "same street words, house differs, name differs": h_both & same_st & (pl.col("house_1") != pl.col("house_2")) & ~nm_eq,
    }
    for name, cond in R.items():
        for s in (2, 3):
            d = a.filter(cond & (pl.col("src") == s))
            n = d.height
            if n == 0:
                continue
            rows.append(dict(country=C, src=s, rule=name, n_pop=int(n / 0.3), pct_of_src=round(100 * n / tot[s], 3),
                             p_true=round(float((d["cls"] == "true").mean()), 4), p_decoy=round(float((d["cls"] == "decoy").mean()), 4)))
    # house edit anatomy: true copies vs decoys with the same name_n and same street, house differs
    h = a.filter(nm_eq & h_both & same_st & (pl.col("house_1") != pl.col("house_2")))
    print(f"\n### {C}: same name + same street + house differs: n={h.height}")
    print(h.group_by("cls", pl.col("hlev").clip(0, 4).alias("lev")).len().pivot(on="cls", index="lev", values="len").sort("lev"))
    print(h.with_columns(pl.col("hdiff").cut([0, 2, 10, 50, 100, 1000]).alias("dbin")).group_by("cls", "dbin").len()
          .pivot(on="cls", index="dbin", values="len").sort("dbin"))
    print(h.with_columns((pl.col("house_2").str.len_chars() - pl.col("house_1").str.len_chars()).alias("dlen"))
          .group_by("cls", "dlen").len().pivot(on="cls", index="dlen", values="len").sort("dlen"))
r = pl.DataFrame(rows)
print(r)
r.write_parquet("/work/dive/g9_rules.parquet")
