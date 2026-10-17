"""Run the real pipeline stages (src/ber/pipeline.py, unchanged) on the demo slice, then export what the page shows.

python demo/run_pipeline.py --data local_data/demo/data --work local_data/demo/work --results local_data/demo/results.json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from ber import pipeline as P  # noqa: E402
from ber.metric import entity_f05  # noqa: E402

REC_COLS = ["entity_id", "src", "country", "business_name", "business_address"]


def stage(name, fn, *args, times):
    print(f"@@stage {name} start", flush=True)
    t = time.time()
    fn(*args)
    times[name] = round(time.time() - t, 1)
    print(f"@@stage {name} done {times[name]}", flush=True)


def rec_info(rec: pl.DataFrame, rid: int) -> dict:
    r = rec.row(rid, named=True)
    return {"id": r["entity_id"], "src": r["src"], "name": r["business_name"], "addr": r["business_address"]}


def export(data: Path, work: Path, times: dict) -> dict:
    rep = json.loads((work / "val_report.json").read_text())
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=REC_COLS)
    lab = P.labels(data, work).with_columns(
        (pl.col("parent").fill_null(pl.col("rid")).hash(P.SEED + 7) % 2 == 0).alias("half_a"))
    truth: dict = {}
    for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
        truth.setdefault(par, set()).add(q)
    s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val")).select("rid", "half_a")
    val_ids = s1v["rid"].to_list()
    best = pl.read_parquet(work / "val_best.parquet").filter(pl.col("s1").is_in(pl.Series(val_ids).implode()))
    cfg = rep["best"]
    chosen = (P.decide_pairs(best, cfg.get("miss", 0.0)) if cfg["mode"] == "efopt"
              else best.filter(pl.col("prob") >= cfg["thr"]).select("q", "s1"))
    sel = set(chosen.iter_rows())
    claim: dict = {}
    for q, s1, prob in best.sort("prob", descending=True).iter_rows():
        claim.setdefault(s1, []).append((q, prob))

    entities = []
    for s1, half_a in s1v.iter_rows():
        t = truth.get(s1, set())
        rows = [{**rec_info(rec, q), "p": round(float(p), 4), "sel": (q, s1) in sel, "true": q in t}
                for q, p in claim.get(s1, [])]
        seen = {q for q, _ in claim.get(s1, [])}
        rows += [{**rec_info(rec, q), "p": None, "sel": False, "true": True} for q in sorted(t - seen)]
        pred = {q for q, _ in claim.get(s1, []) if (q, s1) in sel}
        entities.append({**rec_info(rec, s1), "country": rec["country"][s1], "half": "A" if half_a else "B",
                         "rows": rows, "f05": entity_f05(t, pred)})

    # Dataset tab: a business, its true copies and the look-alikes the blocking step retrieved for it.
    pairs = pl.concat([pl.read_parquet(p, columns=["q", "s1", "score"]) for p in (work / "train").glob("pairs_*.parquet")])
    rich = [e for e in entities if sum(r["true"] for r in e["rows"]) >= 2]
    show_ids = [rec["entity_id"].to_list().index(e["id"]) for e in rich[::max(1, len(rich) // 40)][:40]]
    showcase = []
    for s1 in show_ids:
        c = pairs.filter(pl.col("s1") == s1).sort("score", descending=True).head(10)
        showcase.append({**rec_info(rec, s1), "country": rec["country"][s1],
                         "cands": [{**rec_info(rec, q), "score": round(float(sc), 3), "true": q in truth.get(s1, set())}
                                   for q, _, sc in c.iter_rows()]})

    src_rows = pl.read_parquet(data / "train_source1.parquet").height, *(
        pl.read_parquet(data / f"train_source{s}.parquet").height for s in (2, 3))
    n_pairs = pl.read_parquet(data / "train_pairs.parquet").height
    copies = lab.filter(pl.col("parent").is_not_null()).group_by("parent").len()["len"]
    n_s1_kept = lab.filter(pl.col("src") == 1).height
    hist = [int((copies == k).sum()) for k in range(1, 9)]
    hist = [n_s1_kept - copies.len()] + hist[:7] + [int((copies >= 8).sum())]
    stats = {"s1": src_rows[0], "s2": src_rows[1], "s3": src_rows[2], "true_pairs": n_pairs,
             "s1_kept": n_s1_kept, "hide_pct": P.hide_pct(work), "copies_hist": hist,
             "unmatched": int(lab.filter((pl.col("src") != 1) & pl.col("parent").is_null()).height),
             "queries": int(lab.filter(pl.col("src") != 1).height), "pairs": pairs.height,
             "val_s1": len(val_ids)}
    return {"report": rep, "times": times, "stats": stats, "entities": entities, "showcase": showcase,
            "finished": time.strftime("%Y-%m-%d %H:%M")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    P.NPROC = 4  # ponytail: laptop has little free RAM; each worker process imports polars + numpy
    shutil.rmtree(a.work, ignore_errors=True)
    a.work.mkdir(parents=True)
    (a.work / "config.json").write_text(json.dumps({"hide_pct": 19}))  # same validation design as the full run
    times: dict = {}
    stage("prep", P.prep, a.data, a.work, "train", times=times)
    stage("retrieve", P.retrieve, a.work, "train", times=times)
    stage("features", P.features, a.work, "train", times=times)
    stage("train", P.train, a.data, a.work, times=times)
    print("@@stage export start", flush=True)
    res = export(a.data, a.work, times)
    tmp = a.results.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    tmp.replace(a.results)
    print(f"@@stage export done; macro F0.5 on held-out businesses = {res['report']['val_f05_at_valA_choice']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
