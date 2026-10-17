"""Uncapped name retrieval for empty-address S2/S3 records (local, no AWS).

For every empty-address S2/S3 record: S1s of the same country whose normalised name key (lower, accents stripped, '.'
removed, legal + stop words dropped, token set sorted) equals the record's key (tier 0); if none, S1s at OSA distance 1
on the key string or with the same consonant-skeleton key (tier 1; covers typos and Devanagari transliterations).
Keeps records whose best tier has exactly ONE S1 (namesake ties skipped) and writes features for (record, S1).

  split=train: S1 pool = non-hidden S1s (hide_pct 19, as production); 'retrieved' pairs = local_data/pruned_pairs_train;
               'found' copies of an S1 = its true copies retrieved with it; label y = true parent == candidate.
  split=test : 'retrieved' = output/v28q_fr15/candidate_pairs.tsv; 'found' = fr15 matches.

uv run --no-project --with polars==1.44.2 --with rapidfuzz --with numpy --with unidecode python scripts/unc_build.py train|test
"""
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("POLARS_MAX_THREADS", "6")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "src")
from ber.normalize import name_skel  # noqa: E402

SCR = r"local_data/work"
SEED, HIDE, CAPDF = 42, 19, 3000
CANON = {"ltd": "limited", "limited": "limited", "pvt": "private", "private": "private", "inc": "inc", "incorporated": "inc",
         "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "llc": "llc", "lp": "lp", "llp": "llp", "pc": "pc",
         "pllc": "pllc", "plc": "plc", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
         "snc": "snc", "ei": "ei", "gmbh": "gmbh", "opc": "opc", "pvtltd": "pvtltd"}
STOP = {"the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l", "dba"}
LEGSTOP = sorted(set(CANON) | STOP)
MAXPOS = 60
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def with_keys(df: pl.DataFrame) -> pl.DataFrame:
    t = (pl.col("business_name").fill_null("").str.to_lowercase().str.normalize("NFKD").str.replace_all(r"[^\x00-\x7f]", "")
         .str.replace_all(r"\.", "").str.extract_all(r"[a-z0-9]+"))
    df = df.with_columns(t.alias("_t"))
    core = pl.col("_t").list.eval(pl.element().filter(~pl.element().is_in(LEGSTOP)))
    return df.with_columns(
        core.list.unique().list.sort().list.join(" ").alias("key"),
        core.list.join(" ").alias("seq"),
        pl.col("_t").list.eval(pl.element().filter(pl.element().is_in(list(CANON))).replace_strict(CANON))
          .list.unique().list.sort().list.join(",").alias("legal"),
        pl.col("_t").list.join(" ").alias("full"),
        pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7f]").alias("nonascii"),
    ).drop("_t")


def _skel_chunk(names):
    return [" ".join(sorted(set(name_skel(n).split()))) for n in names]


def skel_keys(names: list[str]) -> list[str]:
    step = max(1, len(names) // 48 + 1)
    with Pool(6) as p:
        return [k for part in p.map(_skel_chunk, [names[i:i + step] for i in range(0, len(names), step)]) for k in part]


def deletions(df: pl.DataFrame, idcol: str) -> pl.DataFrame:
    out = []
    for i in range(MAXPOS):
        d = df.filter(pl.col("key").str.len_chars() > i)
        if d.height == 0:
            break
        out.append(d.select("country", (pl.col("key").str.slice(0, i) + pl.col("key").str.slice(i + 1)).alias("v"), idcol))
    return pl.concat(out).unique()


def main(split: str) -> None:
    from rapidfuzz.distance import OSA
    parts = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f"local_data/{split}_source{k}.parquet").with_columns(pl.lit(k).cast(pl.Int8).alias("src"))
        d = d.with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("emp")).drop("business_address")
        d = with_keys(d)
        if k != 1:   # full name features only for S1 and the empty-address population
            d = d.with_columns([pl.when(pl.col("emp")).then(pl.col(c)).otherwise(None).alias(c) for c in ["business_name", "seq", "legal", "full"]])
        parts.append(d)
    rec = pl.concat(parts).with_row_index("rid")
    del parts
    if split == "train":
        rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
    else:
        rec = rec.with_columns(pl.lit(False).alias("hidden"))
    log("keys", rec.height)
    id2 = rec.select("entity_id", "rid")
    s1 = rec.filter((pl.col("src") == 1) & ~pl.col("hidden")).select("rid", "entity_id", "country", "business_name", "key", "seq", "legal", "full", "nonascii", "emp")
    rr = rec.filter(pl.col("src") != 1)
    # ---- retrieved pairs (q, s1) and found pairs
    if split == "train":
        pp = pl.read_parquet("local_data/pruned_pairs_train.parquet").rename({"s1": "s"})
        gt = pl.read_parquet("local_data/train_pairs.parquet")
        gt = gt.join(id2.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2.rename({"entity_id": "s1", "rid": "s"}), on="s1")
        live = s1.select(pl.col("rid").alias("s"))
        par = gt.join(live, on="s").select("q", pl.col("s").alias("parent"))
        found = par.rename({"parent": "s"}).join(pp, on=["q", "s"])   # true copies retrieved with their S1
        par.group_by(pl.col("parent").alias("s")).agg(pl.len().alias("n_true")).write_parquet(os.path.join(SCR, "unc_train_ntrue.parquet"))
    else:
        cp = pl.read_csv("output/v28q_fr15/candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        cp = cp.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
        pp = cp.join(id2.rename({"entity_id": "candidate_entity_ids", "rid": "q"}), on="candidate_entity_ids") \
               .join(id2.rename({"entity_id": "source1_entity_id", "rid": "s"}), on="source1_entity_id").select("q", "s")
        mr = pl.read_csv("output/v28q_fr15/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        mr = mr.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        found = mr.join(id2.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids") \
                  .join(id2.rename({"entity_id": "source1_entity_id", "rid": "s"}), on="source1_entity_id").select("q", "s")
        del cp, mr
    log("retrieved pairs", pp.height, "found pairs", found.height)
    srcmap = rr.select(pl.col("rid").alias("q"), "src", pl.col("key").alias("qkey"))
    fk = found.join(srcmap, on="q")
    kf = fk.group_by("s").agg(pl.len().alias("k_found"), (pl.col("src") == 2).sum().alias("k2"), (pl.col("src") == 3).sum().alias("k3"))
    kfkey = fk.group_by("s", "qkey").agg(pl.len().alias("k_samekey"))
    s_npp = pp.group_by("s").agg(pl.len().alias("s1_npp"))
    q_npp = pp.group_by("q").agg(pl.len().alias("r_npp"))
    # ---- record-side key frequencies (all S2/S3)
    rkey = rr.filter(pl.col("key") != "").group_by("country", "key").agg(pl.len().alias("n_rec_key"), pl.col("emp").sum().alias("n_rec_key_emp"))
    # ---- S1 token df per country
    tdf = s1.filter(pl.col("key") != "").select("country", pl.col("key").str.split(" ").alias("tok")).explode("tok") \
            .group_by("country", "tok").agg(pl.len().alias("df"))
    # ---- population
    P = rr.filter(pl.col("emp")).select("rid", "entity_id", "country", "src", "business_name", "key", "seq", "legal", "full", "nonascii")
    log("population (empty-address S2/S3)", P.height)
    # tier 0 exact
    g0 = s1.filter(pl.col("key") != "").group_by("country", "key").agg(pl.len().alias("ns0"), pl.col("rid").first().alias("s_exact"))
    P = P.join(g0, on=["country", "key"], how="left").with_columns(pl.col("ns0").fill_null(0))
    # skeleton keys
    s1sk = skel_keys(s1["business_name"].to_list())
    s1 = s1.with_columns(pl.Series("skel", s1sk))
    P = P.with_columns(pl.Series("skel", skel_keys(P["business_name"].to_list())))
    log("skeleton keys done")
    gs = s1.filter(pl.col("skel") != "").group_by("country", "skel").agg(pl.len().alias("n_skel"))
    P = P.join(gs, on=["country", "skel"], how="left").with_columns(pl.col("n_skel").fill_null(0))
    s1small = s1.join(gs.filter(pl.col("n_skel") <= 6), on=["country", "skel"]).select("country", "skel", pl.col("rid").alias("s"), pl.col("key").alias("skey"))
    skp = P.filter((pl.col("skel") != "") & (pl.col("n_skel") <= 6)).select(pl.col("rid").alias("q"), "country", "skel", "key") \
           .join(s1small, on=["country", "skel"]).filter(pl.col("key") != pl.col("skey")).select("q", "s").with_columns(pl.lit(True).alias("is_skel"))
    # edit-1 on the key string (len >= 4)
    Q = P.filter(pl.col("key").str.len_chars() >= 4).select(pl.col("rid").alias("q"), "country", "key")
    S = s1.filter(pl.col("key").str.len_chars() >= 3).select(pl.col("rid").alias("s"), "country", "key")
    qd = deletions(Q, "q")
    qall = pl.concat([Q.select("country", pl.col("key").alias("v"), "q"), qd]).unique()
    e1 = [S.select("country", pl.col("key").alias("v"), "s").join(qd, on=["country", "v"]).select("q", "s")]
    for i in range(MAXPOS):
        d = S.filter(pl.col("key").str.len_chars() > i)
        if d.height == 0:
            break
        dv = d.select("country", (pl.col("key").str.slice(0, i) + pl.col("key").str.slice(i + 1)).alias("v"), "s")
        e1.append(dv.join(qall, on=["country", "v"]).select("q", "s"))
    e1 = pl.concat(e1).unique()
    e1 = e1.join(Q.select("q", pl.col("key").alias("qk")), on="q").join(S.select("s", pl.col("key").alias("sk")), on="s")
    d = np.fromiter((OSA.distance(a, b) for a, b in zip(e1["qk"].to_list(), e1["sk"].to_list())), dtype=np.int16, count=e1.height)
    e1 = e1.with_columns(pl.Series("osa", d)).filter(pl.col("osa") == 1).select("q", "s").with_columns(pl.lit(True).alias("is_ed1"))
    log("edit-1 pairs", e1.height, "skeleton pairs", skp.height)
    near = pl.concat([e1.with_columns(pl.lit(False).alias("is_skel")), skp.with_columns(pl.lit(False).alias("is_ed1"))], how="diagonal") \
             .group_by("q", "s").agg(pl.col("is_ed1").any(), pl.col("is_skel").any())
    nn = near.group_by("q").agg(pl.len().alias("n_near"), pl.col("s").first().alias("s_near"), pl.col("is_ed1").first(), pl.col("is_skel").first())
    P = P.join(nn, left_on="rid", right_on="q", how="left").with_columns(pl.col("n_near").fill_null(0))
    # skeleton groups too large to materialise count as >=2 near S1s
    P = P.with_columns(pl.when((pl.col("n_skel") > 6) & (pl.col("ns0") == 0)).then(pl.max_horizontal("n_near", pl.lit(2))).otherwise(pl.col("n_near")).alias("n_near"))
    # competitor S1s one word away from the record key: +1 token (record dropped a word), -1 token (record appended a
    # word), one-token swap (same token count).  Keys are token sets, so drop-one variants identify these.
    def drop1(df, idcol):
        ids, cty, var, nt = [], [], [], []
        for i, c, k in zip(df[idcol].to_list(), df["country"].to_list(), df["key"].to_list()):
            t = k.split()
            if len(t) < 2:
                continue
            for j in range(len(t)):
                ids.append(i); cty.append(c); var.append(" ".join(t[:j] + t[j + 1:])); nt.append(len(t))
        return pl.DataFrame({idcol: ids, "country": cty, "v": var, "nt": nt}, schema_overrides={idcol: pl.UInt32, "nt": pl.Int32})
    V1 = drop1(s1.filter(pl.col("key") != "").select(pl.col("rid").alias("s"), "country", "key"), "s")
    v1c = V1.group_by("country", "v", "nt").agg(pl.len().alias("cnt"))
    s1c = s1.filter(pl.col("key") != "").group_by("country", "key").agg(pl.len().alias("nex"))
    Pq = P.select(pl.col("rid").alias("q"), "country", "key").filter(pl.col("key") != "")
    R1 = drop1(Pq, "q")
    sup = Pq.join(V1.group_by("country", "v").agg(pl.len().alias("n_super1")), left_on=["country", "key"], right_on=["country", "v"]).select("q", "n_super1")
    mnus = R1.join(s1c, left_on=["country", "v"], right_on=["country", "key"]).group_by("q").agg(pl.col("nex").sum().alias("n_minus1"))
    swp = R1.join(v1c, on=["country", "v", "nt"]).group_by("q").agg(pl.col("cnt").sum().alias("_sw"), pl.col("nt").first().alias("_nt"))
    P = P.join(sup, left_on="rid", right_on="q", how="left").join(mnus, left_on="rid", right_on="q", how="left").join(swp, left_on="rid", right_on="q", how="left")
    P = P.with_columns(pl.col("n_super1").fill_null(0), pl.col("n_minus1").fill_null(0),
                       (pl.col("_sw").fill_null(0) - pl.col("_nt").fill_null(0) * pl.col("ns0")).clip(0).alias("n_swap1")).drop("_sw", "_nt")
    # tier 2: no exact / near S1, exactly one S1 one word away
    el = P.filter((pl.col("ns0") == 0) & (pl.col("n_near") == 0) & ((pl.col("n_super1") + pl.col("n_minus1") + pl.col("n_swap1")) == 1)).select(pl.col("rid").alias("q"), "country", "key")
    r1e = R1.join(el.select("q"), on="q")
    t2 = pl.concat([
        el.join(V1.select("country", "v", "s"), left_on=["country", "key"], right_on=["country", "v"]).select("q", "s", pl.lit("super").alias("kind2")),
        r1e.join(s1.select("country", pl.col("key").alias("v"), pl.col("rid").alias("s")), on=["country", "v"]).select("q", "s", pl.lit("minus").alias("kind2")),
        r1e.join(V1, on=["country", "v", "nt"]).select("q", "s", pl.lit("swap").alias("kind2")),
    ]).unique(subset=["q"], keep="first")
    P = P.join(t2.rename({"s": "s_t2"}), left_on="rid", right_on="q", how="left")
    P = P.with_columns(
        pl.when(pl.col("ns0") >= 1).then(0).when(pl.col("n_near") >= 1).then(1).when(pl.col("s_t2").is_not_null()).then(2).otherwise(None).alias("tier"),
        pl.when(pl.col("ns0") == 1).then(pl.col("s_exact")).when((pl.col("ns0") == 0) & (pl.col("n_near") == 1)).then(pl.col("s_near"))
          .when((pl.col("ns0") == 0) & (pl.col("n_near") == 0)).then(pl.col("s_t2")).otherwise(None).alias("s"))
    # summary of retrieval over the whole population
    log("population by tier / uniqueness:")
    print(P.group_by("country", "tier", (pl.col("s").is_not_null()).alias("unique")).len().sort("country", "tier", "unique"), flush=True)
    C = P.filter(pl.col("s").is_not_null()).rename({"rid": "q"})
    C = C.with_columns(pl.when(pl.col("tier") == 1).then(pl.col("is_ed1")).otherwise(False).fill_null(False).alias("is_ed1"),
                       pl.when(pl.col("tier") == 1).then(pl.col("is_skel")).otherwise(False).fill_null(False).alias("is_skel"))
    sc = s1.select(pl.col("rid").alias("s"), pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s_name"), pl.col("key").alias("s_key"),
                   pl.col("seq").alias("s_seq"), pl.col("legal").alias("s_legal"), pl.col("full").alias("s_full"),
                   pl.col("nonascii").alias("s_nonascii"), pl.col("emp").alias("s_emp"), pl.col("skel").alias("s_skel"))
    C = C.join(sc, on="s", how="left")
    C = C.join(kf, on="s", how="left").join(kfkey.rename({"qkey": "key"}), on=["s", "key"], how="left") \
         .join(s_npp, on="s", how="left").join(q_npp, on="q", how="left") \
         .join(rkey, on=["country", "key"], how="left")
    C = C.with_columns([pl.col(c).fill_null(0) for c in ["k_found", "k2", "k3", "k_samekey", "s1_npp", "r_npp", "n_rec_key", "n_rec_key_emp"]])
    # retrieved flags
    C = C.join(pp.with_columns(pl.lit(True).alias("cand_ret")), on=["q", "s"], how="left").with_columns(pl.col("cand_ret").fill_null(False))
    C = C.join(found.select("q").unique().with_columns(pl.lit(True).alias("assigned")), on="q", how="left").with_columns(pl.col("assigned").fill_null(False))
    # token df features (on the S1 key)
    tk = C.select("q", "country", pl.col("s_key").str.split(" ").alias("tok")).explode("tok").join(tdf, on=["country", "tok"], how="left") \
          .with_columns(pl.col("df").fill_null(0)).group_by("q").agg(pl.col("df").min().alias("df_min"), pl.col("df").max().alias("df_max"),
                                                                      (pl.col("df") + 1).log().mean().alias("df_lmean"), (pl.col("df") > CAPDF).all().alias("allcap"),
                                                                      (pl.col("df") > CAPDF).mean().alias("capfrac"))
    C = C.join(tk, on="q", how="left")
    def legal_mid(full):   # a legal word followed later by a non-legal, non-stop word ("Cardinal LLC Center")
        out = []
        for f in full:
            t = (f or "").split(); seen = False; hit = False
            for w in t:
                if w in CANON:
                    seen = True
                elif seen and w not in STOP:
                    hit = True; break
            out.append(hit)
        return out
    C = C.with_columns(pl.Series("legal_mid", legal_mid(C["full"].to_list())), pl.Series("s_legal_mid", legal_mid(C["s_full"].to_list())))
    if split == "train":
        C = C.join(par, on="q", how="left")
        C = C.join(pp.with_columns(pl.lit(True).alias("par_ret")).rename({"s": "parent"}), on=["q", "parent"], how="left") \
             .with_columns(pl.col("par_ret").fill_null(False), (pl.col("parent") == pl.col("s")).fill_null(False).alias("y"))
    C = C.with_columns([(pl.col("kind2") == k).fill_null(False).alias(f"is_{k}") for k in ["super", "minus", "swap"]])
    C = C.drop([c for c in ["s_exact", "s_near", "s_t2", "kind2"] if c in C.columns])
    out = os.path.join(SCR, f"unc_{split}.parquet")
    C.write_parquet(out)
    log("wrote", out, C.height)


if __name__ == "__main__":
    main(sys.argv[1])
