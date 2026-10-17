"""Per-country pairs / empty S1s of the uploaded submissions and their pairwise differences (LB history analysis).

python scripts/lb_versions.py
"""
import polars as pl

r = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"])
c1 = r.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s"), "country")
FILES = {"v25": "/work/dive/v25_mr.tsv", "v26": "/work/dive/v26_matching_results.tsv", "v26_g2.5": "/work/dive/v26_g2.5_mr.tsv",
         "v28_tl": "/work/dive/v28tl_matching_results.tsv", "v28_final": "/work/dive/v28_final_mr.tsv", "fr7": "/work/out_fr7/matching_results.tsv"}


def load(p):
    g = pl.read_csv(p, separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls() \
            .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s", "matched_entity_ids": "m"}).join(c1, on="s")


D = {k: load(v) for k, v in FILES.items()}
n = c1.group_by("country").len()
for k, d in D.items():
    st = d.group_by("country").agg(pl.len().alias("pairs"), pl.col("s").n_unique().alias("nonempty")).join(n, on="country") \
          .with_columns((pl.col("len") - pl.col("nonempty")).alias("empty")).sort("country")
    print(k, st.select("country", "pairs", "empty").rows())


def diff(a, b):
    A, B = D[a], D[b]
    for c in ("US", "India", "France"):
        x = A.filter(pl.col("country") == c).select("s", "m")
        y = B.filter(pl.col("country") == c).select("s", "m")
        allc = set(c1.filter(pl.col("country") == c)["s"].to_list())
        ea, eb = allc - set(x["s"].to_list()), allc - set(y["s"].to_list())
        rem = x.join(y, on=["s", "m"], how="anti").height
        add = y.join(x, on=["s", "m"], how="anti").height
        print(f"  {a}->{b} {c}: pairs -{rem} +{add}; newly empty {len(eb - ea)}, newly filled {len(ea - eb)}")


for a, b in (("v26", "v26_g2.5"), ("v28_tl", "v28_final"), ("v25", "v26"), ("v26", "v28_tl"), ("v28_final", "fr7")):
    diff(a, b)
