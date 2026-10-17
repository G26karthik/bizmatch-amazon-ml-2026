"""blindA (local, test): cap test for one-token (pseudo / initials) records sitting at a unique-address S1.
If they are copies of that S1, their presence must vanish when the S1's other assigned copies in the same source
already fill the generator cap (S2: 5, S3: 6).  France vs US test."""
import polars as pl, sys
C = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
toks = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref", "matched_entity_ids": "entity_id"})
sp = sp.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))

def run(label, j, s1ids):
    # j: records at the S1's exact address: entity_id, s1, name, s1_name
    j = j.with_columns(toks("name").alias("rt"), toks("s1_name").alias("st"), pl.col("entity_id").str.slice(0, 2).alias("src"))
    j = j.with_columns(((pl.col("rt").list.len() == 1) & ~pl.col("name").str.contains(r"(?i)\.com|www\.|com$|^[@#]")
                        & (pl.col("rt").list.set_intersection(pl.col("st")).list.len() == 0)).alias("pseudo"))
    ps = j.filter(pl.col("pseudo")).select("entity_id", "s1", "src")
    asg = sp.join(s1ids.select(pl.col("s1").alias("ref")), on="ref").join(ps.select("entity_id"), on="entity_id", how="anti")
    c = asg.group_by(["ref", "src"]).len().rename({"ref": "s1", "len": "c_other"})
    base = s1ids.select("s1").join(pl.DataFrame({"src": ["S2", "S3"]}), how="cross")
    base = base.join(c, on=["s1", "src"], how="left").with_columns(pl.col("c_other").fill_null(0))
    pc = ps.group_by(["s1", "src"]).len().rename({"len": "n_pseudo"})
    base = base.join(pc, on=["s1", "src"], how="left").with_columns(pl.col("n_pseudo").fill_null(0))
    ua = ps.join(sp.select("entity_id", pl.lit(True).alias("a")), on="entity_id", how="left").with_columns(pl.col("a").fill_null(False))
    print(f"\n[{label}] one-token pseudo/initial records at unique-address S1s: {ps.height}, assigned share {ua['a'].mean():.3f}")
    pu = ua.filter(~pl.col("a")).group_by(["s1", "src"]).len().rename({"len": "n_un"})
    base = base.join(pu, on=["s1", "src"], how="left").with_columns(pl.col("n_un").fill_null(0))
    t = base.group_by(["src", pl.col("c_other").clip(0, 7)]).agg(pl.len().alias("n_s1"), (pl.col("n_pseudo") > 0).mean().round(4).alias("P_pseudo_present"),
        (pl.col("n_un") > 0).mean().round(5).alias("P_unassigned_pseudo"), (pl.col("n_un") > 0).sum().alias("n_un_s1")).sort(["src", "c_other"])
    print(t)
    # assigned-copies distribution of S1s that have an unassigned pseudo record (k_total incl. both sources)
    kt = sp.join(s1ids.select(pl.col("s1").alias("ref")), on="ref").group_by("ref").len().rename({"ref": "s1", "len": "k_tot"})
    un_s1 = ua.filter(~pl.col("a")).select("s1").unique().join(kt, on="s1", how="left").with_columns(pl.col("k_tot").fill_null(0))
    print("  S1s with an unassigned pseudo record: n", un_s1.height, " predicted-empty:", (un_s1["k_tot"] == 0).sum(), " k_tot dist:", un_s1.group_by(pl.col("k_tot").clip(0, 7)).len().sort("k_tot").to_dicts())

# France
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", "h")
kh = s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
s1 = s1.join(kh, on=["K", "h"]).filter(pl.col("nKH") == 1)
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "name", "K", "h").filter(pl.col("h").is_not_null())
run("France test", recs.join(s1, on=["K", "h"]).select("entity_id", "s1", "name", "s1_name"), s1)
# US test (S1 alone on street key; same-house records cached)
us = pl.read_parquet(f"{C}/test_US_samehouse.parquet").filter(~pl.col("no_house"))
us_s1 = pl.read_parquet(f"{C}/test_US_sig_sub.parquet").select("s1")
run("US test", us.select("entity_id", "s1", "name", "s1_name"), us_s1)
