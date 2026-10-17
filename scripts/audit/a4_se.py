"""Add delta-method standard errors to implied unassigned precision U (q=1)."""
import polars as pl, math
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
NT = {("ns","US"):[900,1229,1816,3375,2896],("ns","India"):[430,595,799,1460,2264],("big","US"):[300,459,592,1661,7278],
      ("big","India"):[250,335,372,666,3610],("small","US"):[120,142,239,559,5760],("small","India"):[160,151,232,433,3587]}
LAB = ["[0.75,0.8)", "[0.8,0.85)", "[0.85,0.9)", "[0.9,0.95)", ">=0.95"]
d = pl.read_parquet("local_data/work/audit/implied.parquet")
tp = {}
for k,f in (("ns","tb_ns_test_top"),("big","tb_big_test_top"),("small","tb_test_top")):
    t = pl.read_parquet(f"local_data/work/{f}.parquet").with_columns(band=pl.col("p").cut([0.75,0.8,0.85,0.9,0.95], labels=["<0.75"]+LAB, left_closed=True).cast(pl.String))
    for c, b, n in t.group_by("country","band").len().iter_rows(): tp[(k,c,b)] = n
rows=[]
for r in d.iter_rows(named=True):
    c2 = "US" if r["country"]=="France" else r["country"]
    nt = NT[(r["model"],c2)][LAB.index(r["band"])]; P, a, A = r["P"], r["a"], r["A"]; ntest = tp[(r["model"], r["country"], r["band"])]
    seP = math.sqrt(P*(1-P)/nt); sea = math.sqrt(a*(1-a)/ntest)
    se = math.sqrt((seP/(1-a))**2 + ((P-A)/(1-a)**2*sea)**2)
    rows.append({**r, "se": se, "U_lo": r["U_q1"]-1.64*se, "U_hi": r["U_q1"]+1.64*se})
o = pl.DataFrame(rows).drop("file").with_columns(pl.col("a","A","U_q1","U_q95","se","U_lo","U_hi").round(3))
print(o); o.write_parquet("local_data/work/audit/implied_se.parquet")
