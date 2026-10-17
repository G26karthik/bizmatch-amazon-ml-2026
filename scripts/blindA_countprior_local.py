"""blindA (local, train truth): can the generator's copy-count distribution P(k) break namesake ties for
empty-address copies?  For an empty-address copy r whose (legal-stripped) name matches >=2 S1s, score each candidate X
by P(c_X+1)/P(c_X) (c_X = X's other true copies) and respect the per-source caps (S2<=5, S3<=6)."""
import polars as pl, numpy as np, sys
R = sys.argv[1]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
LEGAL = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|private|llp|lp|pc|pllc|plc)\b"
def nm(c):
    return pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(LEGAL, " ").str.replace_all(r"\s+", " ").str.strip_chars()
def rd(k):
    return pl.read_csv(f"{R}/train_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
pairs = pairs.with_columns(pl.col("m").str.slice(0, 2).alias("src"))
cnt = pairs.group_by("s1").agg(pl.len().alias("k"), (pl.col("src") == "S2").sum().alias("k2"), (pl.col("src") == "S3").sum().alias("k3"))
s1 = rd(1).select(pl.col("entity_id").alias("s1"), "country", nm("business_name").alias("key")).join(cnt, on="s1", how="left").with_columns(pl.col("k").fill_null(0), pl.col("k2").fill_null(0), pl.col("k3").fill_null(0))
# P(k) from truth
pk = s1.group_by("k").len().sort("k"); P = np.zeros(20);
for k, n in zip(pk["k"].to_list(), pk["len"].to_list()):
    P[k] = n
P = P / P.sum(); P[P == 0] = 1e-9
ratio = {c: P[c + 1] / P[c] for c in range(15)}
print("P(k):", np.round(P[:12], 4).tolist()); print("P(c+1)/P(c):", {c: round(v, 3) for c, v in list(ratio.items())[:10]})
emp = pl.concat([rd(k).filter(pl.col("business_address") == "") for k in [2, 3]]).select(pl.col("entity_id").alias("m"), "country", nm("business_name").alias("key"))
emp = emp.join(pairs.select("m", pl.col("s1").alias("true_s1"), "src"), on="m")
grp = s1.group_by(["country", "key"]).len().rename({"len": "ns"})
emp = emp.join(grp, on=["country", "key"]).filter(pl.col("ns") >= 2)
c = emp.join(s1, on=["country", "key"])  # one row per candidate
c = c.with_columns((pl.col("s1") == pl.col("true_s1")).alias("is_true"))
c = c.with_columns((pl.col("k") - pl.col("is_true").cast(pl.Int64)).alias("c_other"),
                   (pl.col("k2") - (pl.col("is_true") & (pl.col("src") == "S2")).cast(pl.Int64)).alias("c2"),
                   (pl.col("k3") - (pl.col("is_true") & (pl.col("src") == "S3")).cast(pl.Int64)).alias("c3"))
c = c.with_columns(pl.col("c_other").replace_strict(ratio, default=0.0, return_dtype=pl.Float64).alias("score"),
                   (((pl.col("src") == "S2") & (pl.col("c2") >= 5)) | ((pl.col("src") == "S3") & (pl.col("c3") >= 6))).alias("capfull"))
c = c.with_columns(pl.when(pl.col("capfull")).then(0.0).otherwise(pl.col("score")).alias("score_cap"))
has_true = c.group_by("m").agg(pl.col("is_true").any().alias("ht"))
c = c.join(has_true.filter(pl.col("ht")), on="m")
print("ambiguous empty-address copies with true S1 among namesakes:", c["m"].n_unique())
rng = np.random.default_rng(0)
c = c.with_columns(pl.lit(rng.random(c.height)).alias("u"))
def acc(col):
    top = c.sort([col, "u"], descending=[True, True]).group_by("m", maintain_order=True).first()
    return top.group_by(pl.col("ns").clip(2, 6)).agg(pl.len(), pl.col("is_true").mean().round(4).alias("top1")).sort("ns")
print("chance (1/ns):", c.group_by("m").agg(pl.col("ns").first()).group_by(pl.col("ns").clip(2, 6)).agg(pl.len(), (1 / pl.col("ns")).mean().round(4)).sort("ns").to_dicts())
print("random pick:", acc("u").to_dicts())
print("P(k)-ratio prior:", acc("score").to_dicts())
print("P(k)-ratio prior + caps:", acc("score_cap").to_dicts())
# posterior mass on true candidate (normalised scores) -> calibration of the prior as a probability
c = c.with_columns((pl.col("score_cap") / pl.col("score_cap").sum().over("m")).alias("post"))
t = c.filter(pl.col("is_true"))
print("mean prior posterior on the true S1:", round(float(t["post"].mean()), 4), " (uniform:", round(float((1 / t["ns"]).mean()), 4), ")")
print("share of records whose true S1 gets posterior >= 0.8:", round(float((t["post"] >= 0.8).mean()), 4), " >=0.9:", round(float((t["post"] >= 0.9).mean()), 4))
top = c.sort(["post", "u"], descending=[True, True]).group_by("m", maintain_order=True).first()
for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
    s = top.filter(pl.col("post") >= th)
    print(f" prior-only rule: top post >= {th}: n={s.height} precision={s['is_true'].mean():.4f}")
