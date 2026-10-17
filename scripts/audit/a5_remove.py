"""Write remove_pairs.parquet (groups whose implied test precision on unassigned records is likely < 0.76) and a borderline list."""
import polars as pl
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(-1)
FIRM = [("fr43 IN ns .75-.80", "ns", "[0.75,0.8)"), ("fr41 US .80-.85", "ns", "[0.8,0.85)"), ("fr41 US .80-.85", "big", "[0.8,0.85)"),
        ("fr39 IN ns .80-.85", "ns", "[0.8,0.85)"), ("fr35 India band", "small", "[0.8,0.85)"), ("fr42 FR band", "ns", "[0.85,0.9)"),
        ("fr28 small p>=.85", "small", "[0.85,0.9)", "India"), ("fr28 small p>=.85", "small", "[0.9,0.95)", "India")]
BORDER = [("fr35 India band", "big", "[0.8,0.85)"), ("fr40 FR extra", "ns", "[0.9,0.95)"), ("fr28 small p>=.85", "small", "[0.85,0.9)", "US"),
          ("fr38 US ns .85-.90", "ns", "[0.85,0.9)"), ("fr36 ns sel", "ns", "[0.85,0.9)", "India")]
g = pl.read_csv("output/v28q_fr43/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a43 = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", m="matched_entity_ids"))
R = pl.read_parquet("local_data/work/audit/pick_bands.parquet").join(a43, on=["m", "s1"], how="semi")
U = pl.read_parquet("local_data/work/audit/implied_se.parquet")
def sel(spec):
    out = []
    for s in spec:
        step, model, band = s[:3]
        f = (pl.col("step") == step) & (pl.col("model") == model) & (pl.col("band") == band)
        if len(s) > 3: f = f & (pl.col("country") == s[3])
        x = R.filter(f)
        for c in x["country"].unique().to_list():
            u = U.filter((pl.col("step") == step) & (pl.col("model") == model) & (pl.col("band") == band) & (pl.col("country") == c)).row(0, named=True)
            out.append(x.filter(pl.col("country") == c).select("m", "s1", reason=pl.lit(f"{step} | {model} {c} p{band} | implied unassigned precision {u['U_q1']:.2f} (90% {u['U_lo']:.2f}-{u['U_hi']:.2f})")))
    return pl.concat(out).unique(["m", "s1"])
firm, bord = sel(FIRM), sel(BORDER)
firm.write_parquet("local_data/work/audit/remove_pairs.parquet"); bord.write_parquet("local_data/work/audit/remove_pairs_borderline.parquet")
print("FIRM", firm.height); print(firm.group_by("reason").len().sort("reason"))
print("BORDERLINE", bord.height); print(bord.group_by("reason").len().sort("reason"))
