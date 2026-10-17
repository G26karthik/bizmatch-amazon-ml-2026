"""Implied precision of tie-breaker adds on UNASSIGNED test records (selection bias), per pick file / model / country / band.
U = (P_band - a*A*q)/(1-a): P_band = train held-out band precision, a = share of the test band already assigned by the submission,
A = top-pick agreement with the submission on assigned records, q = submission precision on those (1.0 pessimistic, 0.95 lenient)."""
import polars as pl
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
E = [0.75, 0.8, 0.85, 0.9, 0.95]
LAB = ["<0.75", "[0.75,0.8)", "[0.8,0.85)", "[0.85,0.9)", "[0.9,0.95)", ">=0.95"]
P = {  # held-out band precision (logs tb_model*.log, a1_*); France -> US proxy
 "ns": {"US": [None, 0.8188, 0.8487, 0.8882, 0.9455, 0.9762], "India": [None, 0.8294, 0.9008, 0.9387, 0.9678, 0.9837]},
 "big": {"US": [None, 0.78, 0.8257, 0.9206, 0.9416, 0.9869], "India": [None, 0.82, 0.8746, 0.9194, 0.9535, 0.9895]},
 "small": {"US": [None, 0.75, 0.784, 0.876, 0.940, 0.993], "India": [None, 0.78, 0.84, 0.876, 0.940, 0.993]},
}
TOP = {"ns": ("tb_ns_test_top", "fr35_s1"), "big": ("tb_big_test_top", "fr28_s1"), "small": ("tb_test_top", "fr27_s1")}
stats, tops = [], []
for k, (f, c) in TOP.items():
    d = pl.read_parquet(f"local_data/work/{f}.parquet").select("m", "s1", "country", "p", asg=pl.col(c).is_not_null(), agree=pl.col("s1") == pl.col(c))
    d = d.with_columns(band=pl.col("p").cut(E, labels=LAB, left_closed=True).cast(pl.String), model=pl.lit(k))
    tops.append(d.select("m", "s1", "model", "band"))
    stats.append(d.group_by("model", "country", "band").agg(a=pl.col("asg").mean(), A=pl.col("agree").filter(pl.col("asg")).mean(), nband=pl.len()))
st = pl.concat(stats); tp = pl.concat(tops)
FILES = {"fr28 small p>=.85": "agent_tb_adds", "fr34 big p>=.9": "agent_tb_adds_big90", "fr35 India band": "agent_tb_india_band",
         "fr36 ns sel": "agent_tb_namesake_sel", "fr38 US ns .85-.90": "agent_tb_namesake_us8590", "fr39 IN ns .80-.85": "agent_tb_namesake_in8085",
         "fr41 US .80-.85": "agent_tb_us80", "fr43 IN ns .75-.80": "agent_tb_namesake_in75", "fr32 FR p>=.95": "agent_tb_fr95",
         "fr37 FR ns p>=.95": "agent_tb_namesake_fr95", "fr40 FR extra": "agent_tb_fr_extra", "fr42 FR band": "agent_tb_fr_band"}
rows = []
for step, f in FILES.items():
    d = pl.read_parquet(f"local_data/work/{f}.parquet").select("m", "s1", "country").with_columns(step=pl.lit(step), file=pl.lit(f))
    d = d.join(tp.unique(["m", "s1", "model"]), on=["m", "s1"], how="left")
    rows.append(d)
R = pl.concat(rows).join(st, on=["model", "country", "band"], how="left")
def u(model, ctry, band, a, A, q):
    if model is None or band is None or a is None: return None
    pc = P[model]["US" if ctry == "France" else ctry][LAB.index(band)]
    if pc is None: return None
    return (pc - a * (A or 1.0) * q) / (1 - a) if a < 1 else None
out = R.group_by("step", "file", "model", "country", "band").agg(n=pl.len(), a=pl.col("a").first(), A=pl.col("A").first()).sort("step", "model", "country", "band")
out = out.with_columns(P=pl.struct("model", "country", "band").map_elements(lambda s: (P[s["model"]]["US" if s["country"] == "France" else s["country"]][LAB.index(s["band"])] if s["model"] and s["band"] else None), return_dtype=pl.Float64),
                       U_q1=pl.struct("model", "country", "band", "a", "A").map_elements(lambda s: u(s["model"], s["country"], s["band"], s["a"], s["A"], 1.0), return_dtype=pl.Float64),
                       U_q95=pl.struct("model", "country", "band", "a", "A").map_elements(lambda s: u(s["model"], s["country"], s["band"], s["a"], s["A"], 0.95), return_dtype=pl.Float64))
print(out.with_columns(pl.col("a", "A", "U_q1", "U_q95").round(3)))
out.write_parquet("local_data/work/audit/implied.parquet")
R.write_parquet("local_data/work/audit/pick_bands.parquet")
