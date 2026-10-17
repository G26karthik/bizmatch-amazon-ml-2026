"""agent_mt: one-token S2/S3 records at MULTI-TENANT addresses (2-3 S1s format-tolerant equal, d32 address logic).
Builds (record, candidate S1) rows with features; train rows carry truth labels.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_mt/mt_build.py train|test
"""
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "the", "and", "of",
       "inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "company", "lp", "llp", "pc", "l", "d", "s", "a"}
LEGAL_FORM = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "limited", "private", "pvt", "corp",
              "co", "company", "lp", "llp", "pc", "incorporated", "corporation"}
W = "local_data/work/agent_mt"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def nkey_expr(c):
    return (pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[^\w]+", " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter(~pl.element().is_in(list(LEG)) & (pl.element() != ""))).list.join(" "))


def naddr_expr(c):
    return pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[^\w]+", " ").str.strip_chars()


def rel_feats(tok, sname):
    T = toks(sname)
    ws = [w for w in T if w not in LEG]
    ini1 = "".join(w[0] for w in ws) if ws else ""
    ini2 = "".join(w[0] for w in T) if T else ""
    initials = int(len(tok) >= 2 and tok in (ini1, ini2))
    ini_prefix = int(len(tok) >= 2 and (ini1.startswith(tok) or ini2.startswith(tok) or tok.startswith(ini1 or "#")))
    cat = "".join(T)
    catw = "".join(ws)
    pr = fuzz.partial_ratio(tok, cat) if cat else 0
    prw = fuzz.partial_ratio(tok, catw) if catw else 0
    rat = fuzz.ratio(tok, catw) if catw else 0
    handle = int(len(tok) >= 6 and pr >= 90)
    # candidate token that is part of the record token (len>=3), longest such
    sub = max([len(w) for w in ws if len(w) >= 3 and w in tok] or [0])
    # record token prefix shared with a candidate word
    pre = 0
    for w in ws:
        k = 0
        while k < min(len(w), len(tok)) and w[k] == tok[k]:
            k += 1
        pre = max(pre, k)
    first = int(bool(ws) and ws[0][:1] == tok[:1])
    legal = int(any(w in LEGAL_FORM for w in T))
    kind = "initials" if initials else ("handle" if handle else "pseudo")
    return dict(initials=initials, ini_prefix=ini_prefix, pr=pr, prw=prw, rat=rat, handle=handle, sub=sub, sub_frac=sub / max(1, len(tok)),
                pre=pre, first=first, legal=legal, c_ntok=len(T), c_nws=len(ws), c_len=len(catw), kind=kind)


def main(split):
    rec = pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    rec = rec.with_columns(src=pl.col("entity_id").str.slice(0, 2))
    countries = sorted(pl.scan_parquet(f"local_data/{split}_source1.parquet").select(pl.col("country").unique()).collect()["country"].to_list())
    truth = None
    if split == "train":
        tp = pl.read_parquet("local_data/train_pairs.parquet")
        truth = dict(zip(tp["m"].to_list(), tp["s1"].to_list()))
    keep = None
    if split == "test":
        g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t",
                        schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
        keep = g.select(entity_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("entity_id").filter(pl.col("entity_id") != "").unique()
        del g
    rows = []
    for ctry in countries:
        r = rec.filter(pl.col("country") == ctry).collect()
        s1 = r.filter(pl.col("src") == "S1")
        log(ctry, "S1", s1.height)
        dfc = Counter(t for a in s1["business_address"].to_list() for t in set(toks(a)))
        common = {t for t, c in dfc.items() if c > 0.01 * s1.height}

        def parts(a):
            T = toks(a)
            h = sorted(t.lstrip("0") for t in T if t.isdigit())
            st = {t for t in T if not t.isdigit() and t not in common and len(t) >= 3 and t not in ("bis", "ter")}
            return h, st, set(T)
        info, by = {}, defaultdict(list)
        for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
            h, st, T = parts(a)
            info[e] = (n, h, st, T, a)
            if h:
                for w in st:
                    by[(tuple(h), w)].append(e)
        log(ctry, "indexed")
        q = r.filter((pl.col("src") != "S1") & ~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" ")
                     & (pl.col("business_name").fill_null("").str.strip_chars() != "")
                     & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
        if keep is not None:
            q = q.join(keep, on="entity_id", how="anti")
        log(ctry, "one-token queries", q.height)
        groups = []
        for m, n, a, src in q.select("entity_id", "business_name", "business_address", "src").iter_rows():
            h, st, T = parts(a)
            if not h or not st:
                continue
            pool = set()
            for w in st:
                pool.update(by.get((tuple(h), w), ()))
            cands = []
            for s in pool:
                sn, sh, sst, sT, _ = info[s]
                if sh != h or not sst:
                    continue
                js, jf = len(st & sst) / len(st | sst), len(T & sT) / len(T | sT)
                if js >= 0.5 and jf >= 0.35:
                    cands.append((s, js, jf))
            if len(cands) in (2, 3):
                groups.append((m, n, a, src, cands, " ".join(sorted(st)) + "|" + " ".join(h)))
        log(ctry, "multi-tenant groups", len(groups))
        if not groups:
            continue
        cset = {s for g in groups for s, _, _ in g[4]}
        # copies proxy (a): S2/S3 records whose nkey == candidate nkey, per source
        s1c = s1.filter(pl.col("entity_id").is_in(list(cset))).with_columns(nk=nkey_expr("business_name"), na=naddr_expr("business_address"))
        oth = r.filter(pl.col("src") != "S1").select("entity_id", "business_name", "src",
                                                     nk=nkey_expr("business_name"), na=naddr_expr("business_address"))
        nkset = set(s1c["nk"].to_list()) - {""}
        nkc = oth.filter(pl.col("nk").is_in(list(nkset))).group_by("nk", "src").agg(ids=pl.col("entity_id"))
        nkmap = {(k, sr): set(v) for k, sr, v in nkc.iter_rows()}
        # copies proxy (b): same normalized address and name token_set_ratio >= 85
        j = s1c.select("entity_id", "business_name", "na").join(oth.rename({"entity_id": "m", "business_name": "mn"}).select("m", "mn", "src", "na"), on="na")
        sim = [fuzz.token_set_ratio(a_, b_) for a_, b_ in zip(j["business_name"].to_list(), j["mn"].to_list())]
        j = j.with_columns(sim=pl.Series(sim, dtype=pl.Float64))
        addr_all = defaultdict(set)   # all records at same normalized address
        addr_sim = defaultdict(set)
        for e, m_, sr, sm in j.select("entity_id", "m", "src", "sim").iter_rows():
            addr_all[(e, sr)].add(m_)
            if sm >= 85:
                addr_sim[(e, sr)].add(m_)
        nk_of = dict(zip(s1c["entity_id"].to_list(), s1c["nk"].to_list()))
        # one-token query records per candidate address (competition)
        compet = Counter(s for g in groups for s, _, _ in g[4])
        akc = Counter(g[5] for g in groups)
        log(ctry, "proxies done")
        for gi, (m, n, a, src, cands, ak) in enumerate(groups):
            tok = "".join(toks(n))
            other = "S3" if src == "S2" else "S2"
            for s, js, jf in cands:
                sn = info[s][0]
                f = rel_feats(tok, sn)
                nk = nk_of.get(s, "")
                cp = {}
                for sr in ("S2", "S3"):
                    ids = (nkmap.get((nk, sr), set()) if nk else set()) | addr_sim.get((s, sr), set())
                    ids.discard(m)
                    cp[sr] = len(ids)
                al_same = len(addr_all.get((s, src), set()) - {m})
                al_oth = len(addr_all.get((s, other), set()))
                cap_same = 5 if src == "S2" else 6
                rows.append(dict(country=ctry, m=m, s1=s, src=src, n_cand=len(cands), tok_len=len(tok), tok_alpha=int(tok.isalpha()),
                                 tok_digit=int(any(ch.isdigit() for ch in tok)), tok_dot=int("." in (n or "")), js=js, jf=jf,
                                 cp_same=cp[src], cp_oth=cp[other], cp_tot=cp["S2"] + cp["S3"],
                                 cp_same_room=cap_same - cp[src], al_same=al_same, al_oth=al_oth, compet=compet[s],
                                 addr_exact=int(" ".join(toks(a)) == " ".join(toks(info[s][4]))),
                                 label=int(truth.get(m) == s) if truth is not None else -1,
                                 m_has_owner=int(truth.get(m) is not None) if truth is not None else -1,
                                 akey=ak, n_q_addr=akc[ak], mname=n, sname=sn, **f))
            if gi % 5000 == 0:
                log(ctry, gi)
    out = pl.DataFrame(rows)
    out.write_parquet(f"{W}/rows_{split}.parquet")
    log("wrote", out.shape)


if __name__ == "__main__":
    main(sys.argv[1])
