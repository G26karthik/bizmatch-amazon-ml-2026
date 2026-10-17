"""Selection-bias estimate: at each p band, test share assigned by the submission and agreement; implied precision on unassigned
records if the band's train held-out precision held for the whole test band (mixture identity)."""
import polars as pl
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(-1)
for f, col in (("tb_ns_test_top", "fr35_s1"), ("tb_big_test_top", None), ("tb_test_top", None)):
    d = pl.read_parquet(f"local_data/work/{f}.parquet")
    c = col or next(x for x in d.columns if x.startswith("fr") and x.endswith("_s1"))
    print("==", f, c, d.height, [x for x in d.columns if x.startswith("fr")])
    d = d.with_columns(band=pl.col("p").cut([0.75, 0.8, 0.85, 0.9, 0.95], left_closed=True), asg=pl.col(c).is_not_null(),
                       agree=(pl.col("s1") == pl.col(c)))
    print(d.filter(pl.col("p") >= 0.75).group_by("country", "band").agg(n=pl.len(), asg_share=pl.col("asg").mean().round(3),
          agree_on_asg=pl.col("agree").filter(pl.col("asg")).mean().round(4)).sort("country", "band"))
