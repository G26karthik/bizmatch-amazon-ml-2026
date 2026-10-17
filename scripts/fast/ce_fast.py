"""Fast cross-encoder (re)training + scoring, test-like option.

Speed-ups vs kaggle/ce_kernel (random order, tokenise inside the loop, fp16+GradScaler, default attention):
  pre-tokenised once (no padding), length-bucketed batches (pad to batch max of similar lengths), bf16 autocast (no
  scaler), SDPA attention, fused AdamW, pinned-memory async copies, length-sorted big-batch scoring.
Test-like: --drop_h removes pairs whose record is a copy of a 19%-hidden S1 (test has none); --orph_w w upweights
genuine unmatched records (test has ~2x decoys).

python scripts/fast/ce_fast.py --base kaggle/ce_out_base2/ce_model --mode fast --bench 200          # throughput only
python scripts/fast/ce_fast.py --base ... --drop_h --epochs 1 --lr 1e-5 --name tlce --score           # train + score
Scores: local_data/llm/<name>_{val,test}.parquet (q, s1, <name>_logit) for the uncertain zone (unc_{val,test}_text).
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
ap.add_argument("--base", default="kaggle/ce_out_base2/ce_model")
ap.add_argument("--train", default="local_data/fast/tl_train.parquet")
ap.add_argument("--mode", default="fast", choices=["fast", "baseline"])
ap.add_argument("--bench", type=int, default=0, help="only time this many training steps")
ap.add_argument("--n", type=int, default=0, help="subsample training pairs")
ap.add_argument("--drop_h", action="store_true")
ap.add_argument("--orph_w", type=float, default=1.0)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--max_tokens", type=int, default=0, help="token-budget batches: pairs x padded length <= this (long pairs -> smaller batches)")
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--freeze_emb", action="store_true", help="freeze the 250k-token word embeddings (saves ~2.3 GB optimizer state)")
ap.add_argument("--name", default="")
ap.add_argument("--score", action="store_true", help="score unc_{val,test}_text and write local_data/llm/<name>_*.parquet")
ap.add_argument("--score_only", action="store_true")
ap.add_argument("--save", default="")
ap.add_argument("--data_dir", default="local_data/llm")
ap.add_argument("--out_dir", default="local_data/llm")
ap.add_argument("--eval_file", default="kaggle/ce_ds/ce_val.parquet")
a = ap.parse_args()
T0 = time.time()
FAST = a.mode == "fast"
torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
ap_frac = 0.75  # cap the caching allocator below physical VRAM: on Windows WDDM an overflow silently spills to system RAM (10x slower)
if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
    torch.cuda.set_per_process_memory_fraction(ap_frac)


def log(*x):
    print(f"[{time.time() - T0:7.1f}s]", *x, flush=True)


tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, attn_implementation="sdpa" if FAST else "eager").to(dev)
log(f"model {a.base} params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M mode {a.mode} gpu {torch.cuda.get_device_name()}")
PAD = tok.pad_token_id
EVAL_BS = 1024 if torch.cuda.get_device_properties(0).total_memory > 20 * 2**30 else 256
if a.freeze_emb:
    model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
log(f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.0f}M")


def pretok(ta, tb):
    enc = tok(ta, tb, truncation="longest_first", max_length=a.maxlen, padding=False, return_attention_mask=False)
    return [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]


def collate(ids, j):
    L = max(len(ids[k]) for k in j)
    L = -(-L // 16) * 16 if FAST else L  # few distinct shapes -> allocator reuse, tensor-core friendly
    x = np.full((len(j), L), PAD, dtype=np.int64)
    for r, k in enumerate(j):
        x[r, :len(ids[k])] = ids[k]
    xt = torch.from_numpy(x).pin_memory().to(dev, non_blocking=True)
    return xt, (xt != PAD).long()


def bucket_batches(lens, bs, rng):
    idx = rng.permutation(len(lens))
    ch = bs * 64
    out = []
    for i in range(0, len(idx), ch):
        c = idx[i:i + ch]
        c = c[np.argsort(lens[c], kind="stable")]
        if a.max_tokens:
            k = 0
            while k < len(c):
                e = k + 1
                while e < len(c) and e - k < 2 * bs and (e + 1 - k) * (-(-int(lens[c[e]]) // 16) * 16) <= a.max_tokens:
                    e += 1
                out.append(c[k:e]); k = e
        else:
            out += [c[k:k + bs] for k in range(0, len(c), bs)]
    return [out[i] for i in rng.permutation(len(out))]


@torch.inference_mode()
def predict(ta, tb, bs=None):
    bs = bs or (EVAL_BS if FAST else 256)
    model.eval()
    ids = pretok(ta, tb)
    lens = np.array([len(x) for x in ids])
    order = np.argsort(lens)
    out = np.empty(len(ids), np.float32)
    t = time.time()
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        x, m = collate(ids, j)
        with torch.autocast("cuda", dtype=torch.bfloat16 if FAST else torch.float16):
            out[j] = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
    torch.cuda.synchronize()
    log(f"scored {len(ids)} pairs in {time.time() - t:.1f}s ({len(ids) / (time.time() - t):.0f} pairs/s)")
    model.train()
    return out


def eval_ll(n=20000):
    ev = pl.read_parquet(a.eval_file).sample(n, seed=1)
    z = predict(ev["text_a"].to_list(), ev["text_b"].to_list())
    p = 1 / (1 + np.exp(-z.astype(np.float64)))
    y = ev["y"].to_numpy()
    return -np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))


if not a.score_only:
    tr = pl.read_parquet(a.train)
    if a.drop_h:
        tr = tr.filter(~pl.col("is_h"))
    if a.n:
        tr = tr.sample(min(a.n, tr.height), seed=a.seed)
    ta, tb = tr["text_a"].to_list(), tr["text_b"].to_list()
    y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    w = torch.tensor(np.where(tr["is_orph"].to_numpy(), a.orph_w, 1.0), dtype=torch.float32)
    log(f"train pairs {len(ta)} pos {float(y.mean()):.3f} drop_h {a.drop_h} orph_w {a.orph_w}")
    rng = np.random.default_rng(a.seed)
    if FAST:
        t = time.time()
        ids = pretok(ta, tb)
        lens = np.array([len(x) for x in ids])
        log(f"pretokenised in {time.time() - t:.1f}s, mean len {lens.mean():.1f}, p99 {np.quantile(lens, .99):.0f}")
    nep = max(1, math.ceil(a.epochs))
    per_ep = len(bucket_batches(lens, a.bs, np.random.default_rng(0))) if (FAST and a.max_tokens) else math.ceil(len(ta) / a.bs)
    steps = a.bench or int(per_ep * a.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01, fused=FAST)
    sch = get_linear_schedule_with_warmup(opt, min(a.warmup, steps // 10), steps)
    scaler = None if FAST else torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    s, t_start, n_seen, run_loss, pad_tok, real_tok = 0, None, 0, [], 0, 0
    for ep in range(nep):
        if FAST:
            batches = bucket_batches(lens, a.bs, rng)
        else:
            o = rng.permutation(len(ta))
            batches = [o[i:i + a.bs] for i in range(0, len(o), a.bs)]
        for j in batches:
            if s == 20:
                torch.cuda.synchronize(); t_start = time.time(); n_seen = 0
            if FAST:
                x, m = collate(ids, j)
            else:
                enc = tok([ta[k] for k in j], [tb[k] for k in j], truncation="longest_first", max_length=a.maxlen, padding=True, return_tensors="pt")
                x, m = enc["input_ids"].to(dev, non_blocking=True), enc["attention_mask"].to(dev, non_blocking=True)
            pad_tok += x.numel(); real_tok += int(m.sum())
            yy, ww = y[j].to(dev, non_blocking=True), w[j].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16 if FAST else torch.float16):
                logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = (lossf(logit.float(), yy) * ww).sum() / ww.sum()
            opt.zero_grad(set_to_none=True)
            if FAST:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            n_seen += len(j); s += 1
            if s % 200 == 0 and torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
                torch.cuda.empty_cache()  # small WDDM GPU: return fragmented cache so non-PyTorch allocations (cuBLAS, display) fit
            run_loss.append(loss.item() if s % 50 == 0 else None)
            if s % 500 == 0 or (a.bench and s % 50 == 0):
                torch.cuda.synchronize()
                el = time.time() - t_start if t_start else 0
                rl = [v for v in run_loss[-500:] if v is not None]
                log(f"step {s}/{steps} ep {ep} loss~{np.mean(rl):.4f} {n_seen / max(el, 1e-9):.0f} pairs/s pad-eff {real_tok / pad_tok:.2f}")
            if s >= steps:
                break
        if s >= steps:
            break
    torch.cuda.synchronize()
    el = time.time() - t_start
    log(f"TRAIN DONE {s} steps; steady throughput {n_seen / el:.0f} pairs/s; padding efficiency {real_tok / pad_tok:.3f}; peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    if a.save:
        model.save_pretrained(a.save); tok.save_pretrained(a.save); log("saved", a.save)
log(f"eval logloss (ce_val 20k sample, includes H records): {eval_ll():.4f}")
if a.score:
    for part in ("val", "test"):
        d = pl.read_parquet(f"{a.data_dir}/unc_{part}_text.parquet")
        z = predict(d["text_a"].to_list(), d["text_b"].to_list())
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        fn = f"{a.out_dir}/{a.name}_{part}.parquet"
        d.select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).with_columns(pl.Series(f"logit", z)).write_parquet(fn)
        log("wrote", fn, d.height)
log("done")
