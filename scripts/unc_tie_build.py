"""Namesake ties for empty-address S2/S3 records: every (record, namesake S1) row where 2..12 S1s of the country share
the record's exact name key.  Features: legal-form relation + resolver pick, raw-name / word-order equality, namesake
existing matches, caps, count prior P(k+1)/P(k), retrieval flags, production uncertain-zone prob (unc_val / unc_test).

  train: S1 pool = non-hidden S1s (hide 19%), retrieved = pruned_pairs_train, found = true copies retrieved with their S1
  test : retrieved = output/v28q_fr16/candidate_pairs.tsv, found = fr16 matches

uv run --no-project --with polars==1.44.2 --with numpy --with unidecode --with rapidfuzz python scripts/unc_tie_build.py train|test
"""
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "6")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "scripts")
from unc_build import CANON, HIDE, SCR, SEED, STOP, log, with_keys  # noqa: E402

SUB = "output/v28q_fr16"
MAXNS = 12


def legal_mid(full):
    out = []
    for f in full:
        seen = hit = False
        for w in (f or "").split():
            if w in CANON:
                seen = True
            elif seen and w not in STOP:
                hit = True
                break
        out.append(hit)
    return out


def main(split):
    parts = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f"local_data/{split}_source{k}.parquet").with_columns(pl.lit(k).cast(pl.Int8).alias("src"))
        d = d.with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("emp")).drop("business_address")
        d = with_keys(d)
        if k != 1:
            d = d.with_columns([pl.when(pl.col("emp")).then(pl.col(c)).otherwise(None).alias(c) for c in ["business_name", "seq", "legal", "full"]])
        parts.append(d)
    rec = pl.concat(parts).with_row_index("rid")
    del parts
    if split == "train":
        rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
    else:
        rec = rec.with_columns(pl.lit(False).alias("hidden"))
    id2 = rec.select("entity_id", "rid")
    s1 = rec.filter((pl.col("src") == 1) & ~pl.col("hidden")).select("rid", "entity_id", "country", "business_name", "key", "seq", "legal", "full", "emp")
    rr = rec.filter(pl.col("src") != 1)
    if split == "train":
        pp = pl.read_parquet("local_data/pruned_pairs_train.parquet").rename({"s1": "s"})
        gt = pl.read_parquet("local_data/train_pairs.parquet")
        gt = gt.join(id2.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2.rename({"entity_id": "s1", "rid": "s"}), on="s1")
        par = gt.join(s1.select(pl.col("rid").alias("s")), on="s").select("q", pl.col("s").alias("parent"))
        found = par.rename({"parent": "s"}).join(pp, on=["q", "s"])
        unc = pl.read_parquet("local_data/llm/unc_val.parquet")
        ids1 = rec.filter(pl.col("src") == 1).select("rid", (pl.col("entity_id").hash(SEED) % 10 == 0).alias("own_val"))
    else:
        cp = pl.read_csv(f"{SUB}/candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        cp = cp.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
        pp = cp.join(id2.rename({"entity_id": "candidate_entity_ids", "rid": "q"}), on="candidate_entity_ids") \
               .join(id2.rename({"entity_id": "source1_entity_id", "rid": "s"}), on="source1_entity_id").select("q", "s")
        mr = pl.read_csv(f"{SUB}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        mr = mr.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        found = mr.join(id2.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids") \
                  .join(id2.rename({"entity_id": "source1_entity_id", "rid": "s"}), on="source1_entity_id").select("q", "s")
        unc = pl.read_parquet("local_data/llm/unc_test.parquet")
        del cp, mr
    unc = unc.select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32).alias("s"), pl.col("prob").cast(pl.Float64).alias("uprob"))
    log("retrieved", pp.height, "found", found.height)
    srcmap = rr.select(pl.col("rid").alias("q"), "src")
    kf = found.join(srcmap, on="q").group_by("s").agg(pl.len().alias("k_found"), (pl.col("src") == 2).sum().alias("k2"), (pl.col("src") == 3).sum().alias("k3"))
    s_npp = pp.group_by("s").agg(pl.len().alias("s1_npp"))
    q_npp = pp.group_by("q").agg(pl.len().alias("r_npp"))
    rkey = rr.filter(pl.col("key") != "").group_by("country", "key").agg(pl.len().alias("n_rec_key"), pl.col("emp").sum().alias("n_rec_key_emp"))
    tdf = s1.filter(pl.col("key") != "").select("country", pl.col("key").str.split(" ").alias("tok")).explode("tok").group_by("country", "tok").agg(pl.len().alias("df"))
    # count prior from train truth (copies per S1 incl. zero)
    tp = pl.read_parquet("local_data/train_pairs.parquet").group_by("s1").len()
    n1 = pl.scan_parquet("local_data/train_source1.parquet").select(pl.len()).collect().item()
    cnt = np.bincount(np.minimum(tp["len"].to_numpy(), 30), minlength=32).astype(float)
    cnt[0] += n1 - tp.height
    P = cnt / cnt.sum(); P[P == 0] = 1e-9
    ratio = {c: float(P[c + 1] / P[c]) for c in range(31)}
    # population: empty-address records whose exact key is shared by 2..MAXNS S1s
    E = rr.filter(pl.col("emp") & (pl.col("key") != "")).select(pl.col("rid").alias("q"), pl.col("entity_id").alias("m_id"), "country", "src", "business_name", "key", "seq", "legal", "full", "nonascii")
    g0 = s1.filter(pl.col("key") != "").group_by("country", "key").agg(pl.len().alias("ns0"))
    E = E.join(g0, on=["country", "key"]).filter((pl.col("ns0") >= 2) & (pl.col("ns0") <= MAXNS))
    log("tie records", E.height)
    sc = s1.select(pl.col("rid").alias("s"), pl.col("entity_id").alias("s1_id"), "country", "key", pl.col("business_name").alias("s_name"),
                   pl.col("seq").alias("s_seq"), pl.col("legal").alias("s_legal"), pl.col("full").alias("s_full"), pl.col("emp").alias("s_emp"))
    R = E.join(sc, on=["country", "key"])
    R = R.join(kf, on="s", how="left").join(s_npp, on="s", how="left").join(q_npp, on="q", how="left").join(rkey, on=["country", "key"], how="left")
    R = R.with_columns([pl.col(c).fill_null(0) for c in ["k_found", "k2", "k3", "s1_npp", "r_npp", "n_rec_key", "n_rec_key_emp"]])
    R = R.join(pp.with_columns(pl.lit(True).alias("ret")), on=["q", "s"], how="left").with_columns(pl.col("ret").fill_null(False))
    R = R.join(found.with_columns(pl.lit(True).alias("is_found")), on=["q", "s"], how="left").with_columns(pl.col("is_found").fill_null(False))
    R = R.join(found.select("q").unique().with_columns(pl.lit(True).alias("assigned")), on="q", how="left").with_columns(pl.col("assigned").fill_null(False))
    R = R.join(unc, on=["q", "s"], how="left").join(unc.group_by("q").agg(pl.col("uprob").max().alias("q_uprob")), on="q", how="left")
    # relations
    rel = []
    for a, b in zip(R["legal"].to_list(), R["s_legal"].to_list()):
        ra, rb = set(filter(None, (a or "").split(","))), set(filter(None, (b or "").split(",")))
        rel.append(1 if ra == rb else 2 if ra < rb else 3 if rb < ra else 4)   # 1 same (incl. both empty), 2 record drops, 3 adds, 4 swap
    norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]", "")  # noqa: E731
    R = R.with_columns(pl.Series("legal_rel", rel, dtype=pl.Int8),
                       (pl.col("business_name") == pl.col("s_name")).alias("raw_eq"),
                       (norm("business_name") == norm("s_name")).alias("rawnorm_eq"),
                       (pl.col("full") == pl.col("s_full")).alias("full_eq"),
                       (pl.col("seq") == pl.col("s_seq")).alias("seq_eq"),
                       ((pl.col("legal") == "") & (pl.col("s_legal") == "")).alias("both_nolegal"),
                       (pl.col("legal") != "").alias("r_has_legal"), (pl.col("s_legal") != "").alias("s_has_legal"),
                       (((pl.col("src") == 2) & (pl.col("k2") >= 5)) | ((pl.col("src") == 3) & (pl.col("k3") >= 6))).alias("cap_full"),
                       pl.col("k_found").clip(0, 30).replace_strict(ratio, default=0.0, return_dtype=pl.Float64).alias("prior"),
                       pl.Series("legal_mid", legal_mid(R["full"].to_list())), pl.Series("s_legal_mid", legal_mid(R["s_full"].to_list())),
                       pl.col("key").str.split(" ").list.len().alias("key_ntok"), pl.col("key").str.len_chars().alias("key_len"),
                       pl.col("business_name").str.len_chars().alias("name_len"), pl.col("s_name").str.len_chars().alias("s_name_len"))
    R = R.with_columns(pl.when(pl.col("cap_full")).then(0.0).otherwise(pl.col("prior")).alias("prior_cap"))
    w = pl.col("q")
    R = R.with_columns(
        (pl.col("legal_rel") == 1).sum().over(w).alias("n_same"),
        pl.col("legal_rel").is_in([2, 3]).sum().over(w).alias("n_dropadd"),
        pl.col("rawnorm_eq").sum().over(w).alias("n_rawnorm"), pl.col("seq_eq").sum().over(w).alias("n_seq"),
        pl.col("full_eq").sum().over(w).alias("n_full"),
        pl.col("ret").sum().over(w).alias("n_ret"), pl.col("cap_full").sum().over(w).alias("n_capfull"),
        (pl.col("prior_cap") / pl.col("prior_cap").sum().over(w)).fill_nan(0.0).alias("post"),
        pl.col("k_found").rank("min", descending=True).over(w).alias("k_rank"),
        (pl.col("k_found") - pl.col("k_found").max().over(w)).alias("k_minus_max"),
        pl.col("k_found").mean().over(w).alias("k_mean"),
        (pl.col("s1_npp") - pl.col("s1_npp").max().over(w)).alias("npp_minus_max"),
    )
    R = R.with_columns(
        pl.when((pl.col("legal_rel") == 1) & (pl.col("n_same") == 1)).then(pl.lit("same"))
          .when((pl.col("n_same") == 0) & pl.col("legal_rel").is_in([2, 3]) & (pl.col("n_dropadd") == 1)).then(pl.lit("dropadd"))
          .otherwise(None).alias("pick_cue"))
    R = R.with_columns(pl.col("pick_cue").is_not_null().alias("is_pick"),
                       pl.col("pick_cue").is_not_null().any().over(w).alias("q_has_pick"),
                       pl.col("rawnorm_eq").cast(pl.Int8).alias("_r"))
    R = R.with_columns(((pl.col("n_rawnorm") == 1) & pl.col("rawnorm_eq")).alias("uniq_rawnorm"), ((pl.col("n_seq") == 1) & pl.col("seq_eq")).alias("uniq_seq")).drop("_r")
    tk = E.select("q", "country", pl.col("key").str.split(" ").alias("tok")).explode("tok").join(tdf, on=["country", "tok"], how="left") \
          .with_columns(pl.col("df").fill_null(0)).group_by("q").agg(pl.col("df").min().alias("df_min"), (pl.col("df") + 1).log().mean().alias("df_lmean"),
                                                                      (pl.col("df") > 3000).all().alias("allcap"))
    R = R.join(tk, on="q", how="left")
    if split == "train":
        R = R.join(par, on="q", how="left")
        R = R.join(pp.with_columns(pl.lit(True).alias("par_ret")).rename({"s": "parent"}), on=["q", "parent"], how="left") \
             .with_columns(pl.col("par_ret").fill_null(False), (pl.col("parent") == pl.col("s")).fill_null(False).alias("y"))
        R = R.join(ids1.rename({"rid": "parent", "own_val": "par_val"}), on="parent", how="left") \
             .join(rec.select(pl.col("rid").alias("q"), (pl.col("entity_id").hash(SEED) % 10 == 0).alias("q_own_val")), on="q", how="left") \
             .with_columns(pl.coalesce("par_val", "q_own_val").alias("is_val")).drop("par_val", "q_own_val")
    R = R.drop("seq", "full", "s_seq", "s_full")
    out = os.path.join(SCR, f"tie_{split}.parquet")
    R.write_parquet(out)
    log("wrote", out, R.height, "rows,", R["q"].n_unique(), "records")


if __name__ == "__main__":
    main(sys.argv[1])
