"""Generator dive 4 (worker): cross-record structure within S1 groups.
 - copies per S1 per source (all S1s): distribution, joint S2 x S3, independence check
 - shared noise: are two copies of one S1 closer to each other than to the S1 (common noisy intermediate)?
 - decoys (raw-unmatched, best full-pool candidate = S1 X): per-S1 counts, tie to a specific copy
needs tp.parquet, best2.parquet (g2_prep.py)
"""
import itertools

import numpy as np
import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70)
R = "/work/run4/train/records.parquet"
ids = pl.scan_parquet(R).select("src", "country").with_row_index("rid").filter(pl.col("src") == 1).collect()
tp = pl.read_parquet("/work/dive/tp.parquet").with_columns(pl.col("s1r").cast(pl.UInt32), pl.col("mr").cast(pl.UInt32))
cnt = tp.group_by("s1r").agg((pl.col("msrc") == 2).sum().alias("n2"), (pl.col("msrc") == 3).sum().alias("n3"))
s1 = ids.rename({"rid": "s1r"}).join(cnt, on="s1r", how="left").fill_null(0)

# decoys attached to each S1 (best full-pool candidate of raw-unmatched records), by strength
b = pl.read_parquet("/work/dive/best2.parquet")
dec = b.filter(~pl.col("matched") & pl.col("s1").is_not_null())
dec_strong = dec.filter(pl.col("pA") >= 0.5)
dcnt = dec.group_by("s1").agg(pl.len().alias("nd_any"), (pl.col("pA") >= 0.5).sum().alias("nd_strong"),
                              (pl.col("pA") >= 0.1).sum().alias("nd_p01"))
s1 = s1.join(dcnt.rename({"s1": "s1r"}), on="s1r", how="left").fill_null(0)
s1.write_parquet("/work/dive/s1_counts.parquet")

for c in ("US", "India"):
    d = s1.filter(pl.col("country") == c)
    print(f"\n##### {c}: S1 entities {d.height}")
    for col in ("n2", "n3", "nd_any", "nd_p01", "nd_strong"):
        vc = d.group_by(col).len().sort(col)
        print(col, dict(zip(vc[col].to_list(), vc["len"].to_list())))
    tot = (d["n2"] + d["n3"])
    vc = tot.value_counts().sort(tot.name)
    print("n2+n3", dict(zip(vc[tot.name].to_list(), vc["count"].to_list())))
    print("corr(n2,n3) =", round(float(np.corrcoef(d["n2"].to_numpy(), d["n3"].to_numpy())[0, 1]), 4),
          " corr(n2+n3, nd_strong) =", round(float(np.corrcoef((d["n2"] + d["n3"]).to_numpy(), d["nd_strong"].to_numpy())[0, 1]), 4))
    # joint vs independence (observed / expected) for n2, n3 in 0..6
    ct = d.group_by("n2", "n3").len()
    p2 = d.group_by("n2").len().with_columns((pl.col("len") / d.height).alias("p2")).drop("len")
    p3 = d.group_by("n3").len().with_columns((pl.col("len") / d.height).alias("p3")).drop("len")
    ct = ct.join(p2, on="n2").join(p3, on="n3").with_columns((pl.col("len") / (pl.col("p2") * pl.col("p3") * d.height)).round(2).alias("o_e"))
    print("joint n2 x n3 counts:")
    print(ct.pivot(on="n3", index="n2", values="len", sort_columns=True).sort("n2").fill_null(0))
    print("observed/expected under independence:")
    print(ct.pivot(on="n3", index="n2", values="o_e", sort_columns=True).sort("n2").fill_null(0))
    print("strong decoys (pA>=0.5) by total copies n2+n3:")
    print(d.with_columns((pl.col("n2") + pl.col("n3")).clip(0, 9).alias("ncop")).group_by("ncop").agg(
        pl.len(), pl.col("nd_strong").mean().round(4).alias("mean_dec"), (pl.col("nd_strong") > 0).mean().round(4).alias("p_has_dec")).sort("ncop"))

# ---------------- shared noise & decoy-copy ties on a 1/40 sample of S1 groups
samp = s1.filter(pl.col("s1r").hash(5) % 40 == 0)
cop = tp.join(samp.select("s1r"), on="s1r").select("s1r", pl.col("mr").alias("r"), pl.col("msrc").alias("src"), pl.lit("c").alias("kind"))
dd = dec.join(samp.select(pl.col("s1r").alias("s1")), on="s1").select(pl.col("s1").alias("s1r"), pl.col("q").alias("r"), "src", pl.lit("d").alias("kind"), "pA")
own = samp.select("s1r", pl.col("s1r").alias("r"), pl.lit(1, pl.Int8).alias("src"), pl.lit("s").alias("kind"))
g = pl.concat([own.with_columns(pl.lit(None, pl.Float32).alias("pA")), cop.with_columns(pl.lit(None, pl.Float32).alias("pA")),
               dd.with_columns(pl.col("pA").cast(pl.Float32))], how="diagonal_relaxed")
txt = pl.scan_parquet(R).select("business_name", "business_address", "name_n", "addr_n", "house", "country").with_row_index("rid") \
        .join(g.select(pl.col("r").alias("rid")).lazy().unique(), on="rid").collect()
g = g.join(txt.rename({"rid": "r"}), on="r")
g.write_parquet("/work/dive/groups_sample.parquet")
print("\ngroup sample rows", g.height, g.group_by("kind").len())


def bare(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


rows = []
for (s1r,), grp in g.group_by(["s1r"]):
    S = grp.filter(pl.col("kind") == "s").row(0, named=True)
    C = grp.filter(pl.col("kind") == "c").to_dicts()
    D = grp.filter(pl.col("kind") == "d").to_dicts()
    for a, bb in itertools.combinations(C, 2):
        # name on bare lower text (case/punct neutral), address on normalised addr_n
        na, nb, ns = bare(a["business_name"]), bare(bb["business_name"]), bare(S["business_name"])
        aa, ab, as_ = a["addr_n"] or "", bb["addr_n"] or "", S["addr_n"] or ""
        rows.append(dict(kind="cc", country=S["country"], pair=f"S{a['src']}S{bb['src']}",
                         n_eq=na == nb, n_both_noisy=(na != ns) and (nb != ns),
                         n_cc=fuzz.ratio(na, nb), n_cs=max(fuzz.ratio(na, ns), fuzz.ratio(nb, ns)),
                         a_both_nonempty=bool(aa and ab and as_), a_eq=aa == ab, a_both_noisy=(aa != as_) and (ab != as_),
                         a_cc=fuzz.ratio(aa, ab), a_cs=max(fuzz.ratio(aa, as_), fuzz.ratio(ab, as_)),
                         h_both_changed_same=(a["house"] == bb["house"]) and a["house"] != S["house"] and bool(a["house"])))
    for d in D:
        best_c_addr = max([fuzz.ratio(d["addr_n"] or "", c["addr_n"] or "") for c in C], default=-1)
        best_c_name = max([fuzz.ratio(bare(d["business_name"]), bare(c["business_name"])) for c in C], default=-1)
        rows.append(dict(kind="dc", country=S["country"], pair=f"D{d['src']}", pA=d["pA"], ncop=len(C),
                         d_addr_empty=not (d["addr_n"] or ""), s_addr_empty=not (S["addr_n"] or ""),
                         a_ds=fuzz.ratio(d["addr_n"] or "", S["addr_n"] or ""), a_dc=best_c_addr,
                         n_ds=fuzz.ratio(bare(d["business_name"]), bare(S["business_name"])), n_dc=best_c_name,
                         h_eq_s=(d["house"] == S["house"]) and bool(S["house"]),
                         h_eq_any_copy=any(d["house"] == c["house"] and bool(c["house"]) for c in C),
                         h_eq_copy_not_s=any(d["house"] == c["house"] and bool(c["house"]) for c in C) and d["house"] != S["house"]))
cc = pl.DataFrame([x for x in rows if x["kind"] == "cc"], infer_schema_length=None)
dc = pl.DataFrame([x for x in rows if x["kind"] == "dc"], infer_schema_length=None)
cc.write_parquet("/work/dive/g4_cc.parquet"); dc.write_parquet("/work/dive/g4_dc.parquet")
print("\n=== copy-copy pairs (same S1) ===")
print(cc.group_by("country", "pair").agg(
    pl.len(),
    pl.col("n_eq").mean().round(4).alias("name_eq"),
    pl.col("n_eq").filter(pl.col("n_both_noisy")).mean().round(4).alias("name_eq|both_noisy"),
    (pl.col("n_cc") > pl.col("n_cs")).filter(pl.col("n_both_noisy")).mean().round(4).alias("closer_to_each_other|noisy"),
    pl.col("a_eq").filter(pl.col("a_both_nonempty")).mean().round(4).alias("addr_eq"),
    pl.col("a_eq").filter(pl.col("a_both_nonempty") & pl.col("a_both_noisy")).mean().round(4).alias("addr_eq|both_noisy"),
    (pl.col("a_cc") > pl.col("a_cs")).filter(pl.col("a_both_nonempty") & pl.col("a_both_noisy")).mean().round(4).alias("a_closer_each_other"),
    pl.col("h_both_changed_same").mean().round(4).alias("same_wrong_house"),
).sort("country", "pair"))
print("\n=== decoy vs S1 and vs S1's copies (decoys with best candidate in sample) ===")
print(dc.group_by("country", pl.col("pA") >= 0.5).agg(
    pl.len(), pl.col("ncop").mean().round(2), (pl.col("ncop") > 0).mean().round(4).alias("s1_has_copy"),
    pl.col("d_addr_empty").mean().round(4), pl.col("h_eq_s").mean().round(4),
    pl.col("h_eq_any_copy").mean().round(4), pl.col("h_eq_copy_not_s").mean().round(4),
    (pl.col("a_dc") > pl.col("a_ds")).filter(~pl.col("d_addr_empty") & (pl.col("ncop") > 0)).mean().round(4).alias("addr_closer_to_a_copy"),
    (pl.col("n_dc") > pl.col("n_ds")).filter(pl.col("ncop") > 0).mean().round(4).alias("name_closer_to_a_copy"),
).sort("country", "pA"))
