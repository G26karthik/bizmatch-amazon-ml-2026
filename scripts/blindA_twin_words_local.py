"""blindA (local, US train): singleton 'twin clusters' vs ordinary decoys: which words are added?
Uses cached us_join.parquet (records on the S1's street key; unmatched or true copies)."""
import polars as pl, sys, collections
C = sys.argv[1]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(240)
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
s1 = pl.read_parquet(f"{C}/us_s1.parquet")
kc = s1.group_by("K").len().rename({"len": "nK"})
j = pl.read_parquet(f"{C}/us_join.parquet").join(kc, on="K").filter(pl.col("nK") == 1)
j = j.with_columns(norm("name").alias("rn"), norm("s1_name").alias("sn"))
j = j.with_columns(pl.col("rn").str.split(" ").list.set_difference(pl.col("sn").str.split(" ")).alias("addw"),
                   pl.col("sn").str.split(" ").list.set_difference(pl.col("rn").str.split(" ")).alias("dropw"))
cop = j.filter((pl.col("true_s1") == pl.col("s1")) & (pl.col("d") == 0))
off = j.filter(pl.col("true_s1").is_null() & pl.col("d").is_in(OFF))
cs = off.group_by(["s1", "h"]).len().rename({"len": "cs"})
off = off.join(cs, on=["s1", "h"])
off = off.with_columns(pl.when(pl.col("k") == 0).then(pl.lit("single")).otherwise(pl.lit("nonsingle")).alias("grp"))
print("offset records:", off.group_by(["grp", "cs"]).len().sort(["grp", "cs"]).to_dicts())
def wc(df):
    c = collections.Counter()
    for ws in df["addw"].to_list():
        c.update(set(ws))
    return c
cc = wc(cop); cs_ = wc(off.filter(pl.col("grp") == "single")); cn = wc(off.filter(pl.col("grp") == "nonsingle"))
ncop, ns, nn = cop.height, off.filter(pl.col("grp") == "single").height, off.filter(pl.col("grp") == "nonsingle").height
print(f"copies {ncop}, singleton-offset recs {ns}, nonsingleton-offset recs {nn}")
words = set(w for w, _ in cn.most_common(60)) | set(w for w, _ in cs_.most_common(60)) | set(w for w, _ in cc.most_common(30))
rows = [(w, cc[w] / ncop, cs_[w] / ns, cn[w] / nn) for w in words]
rows.sort(key=lambda r: -(r[2] + r[3]))
print(f"{'word':15s} {'copy%':>8s} {'singOff%':>9s} {'nonsOff%':>9s}")
for w, a, b, c in rows[:70]:
    print(f"{w:15s} {100*a:8.3f} {100*b:9.3f} {100*c:9.3f}")
# name relation classes
def cls(df):
    return df.with_columns(
        (pl.col("rn") == pl.col("sn")).alias("identical"),
        (pl.col("addw").list.len() == 0).alias("no_add"),
        (pl.col("dropw").list.len() == 0).alias("no_drop"))
for nm, df in [("copies d=0", cop), ("single-offset", off.filter(pl.col("grp") == "single")), ("nonsingle-offset", off.filter(pl.col("grp") == "nonsingle"))]:
    d = cls(df)
    print(nm, d.select(pl.len(), pl.col("identical").mean().round(4), pl.col("no_add").mean().round(4), pl.col("no_drop").mean().round(4),
                        (pl.col("addw").list.len()).mean().round(3).alias("mean_add"), (pl.col("dropw").list.len()).mean().round(3).alias("mean_drop")).to_dicts())
off.select("s1", "h", "cs", "grp", "entity_id", "name", "s1_name", "addw", "dropw", "d").write_parquet(f"{C}/us_train_offset_recs.parquet")
cop.select("s1", "entity_id", "name", "s1_name", "addw", "dropw").write_parquet(f"{C}/us_train_copies_d0.parquet")
