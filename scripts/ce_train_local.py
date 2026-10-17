"""Train a cross-encoder on a local GPU (same recipe as the Kaggle jobs) and score the evaluation files.

python scripts/ce_train_local.py --base intfloat/multilingual-e5-small --train kaggle/ce3_ds/ce_train.parquet \
    --epochs 2 --lr 5e-5 --bs 128 --out local_data/ce_small2 \
    --score ce_scores_val=kaggle/ce3_ds/ce_val.parquet ce_extra_scores_val=kaggle/ce_extra_ds/ce_val.parquet \
            ce_scores_test=kaggle/ce3_ds/ce_test.parquet ce_extra_scores_test=kaggle/ce_extra_ds/ce_test.parquet \
            fr_new=local_data/fr_ce/fr_ce_texts.parquet
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--train", type=Path, required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--bs_eval", type=int, default=1024)
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--score", nargs="*", default=[])
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
T0 = time.time()
torch.manual_seed(a.seed)


def log(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


tok = AutoTokenizer.from_pretrained(a.base)
model_dir = a.out / "ce_model"
if (model_dir / "config.json").exists():
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()
    log("loaded trained model", model_dir)
else:
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).cuda()
    tr = pl.read_parquet(a.train)
    ta, tb = tr["text_a"].to_list(), tr["text_b"].to_list()
    y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    rng = np.random.default_rng(a.seed)
    lens = np.array([len(x) + len(z) for x, z in zip(ta, tb)])
    perm = rng.permutation(len(ta))
    chunk, bl = a.bs * 64, []
    for i in range(0, len(perm), chunk):
        c = perm[i:i + chunk]
        c = c[np.argsort(lens[c], kind="stable")]
        bl += [c[k:k + a.bs] for k in range(0, len(c), a.bs)]
    batches = [bl[i] for _ in range(a.epochs) for i in rng.permutation(len(bl))]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, 1000, len(batches))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    log(f"train pairs {len(ta)}, steps {len(batches)}")
    for s, j in enumerate(batches):
        enc = tok([ta[k] for k in j], [tb[k] for k in j], truncation="longest_first", max_length=a.maxlen,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**enc).logits.squeeze(-1).float(), y[j].cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if s % 1000 == 0:
            log(f"step {s}/{len(batches)} loss {loss.item():.4f}")
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
model.eval()
for spec in a.score:
    name, path = spec.split("=", 1)
    if (a.out / f"{name}.parquet").exists():
        continue
    d = pl.read_parquet(path)
    ta, tb = d["text_a"].to_list(), d["text_b"].to_list()
    order = np.argsort([len(x) + len(z) for x, z in zip(ta, tb)])
    p = np.empty(len(ta), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), a.bs_eval):
            j = order[i:i + a.bs_eval]
            enc = tok([ta[k] for k in j], [tb[k] for k in j], truncation="longest_first", max_length=a.maxlen,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                p[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    out = d.select("q", "s1").with_columns(pl.Series("pce", p))
    if "y" in d.columns:
        yy = d["y"].to_numpy()
        log(f"{name}: logloss {-np.mean(yy * np.log(p + 1e-7) + (1 - yy) * np.log(1 - p + 1e-7)):.4f}")
    out.write_parquet(a.out / f"{name}.parquet")
    log(f"scored {name}: {len(ta)}")
log("done")
