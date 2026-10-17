"""LoRA match judge on a Lightning GPU studio.

Qwen2.5-7B (Apache-2.0, 7.6B params) with a 1-logit sequence-classification head, LoRA (r 16) on all attention and MLP
projections, fine-tuned on training pairs from the uncertain zone (ce_train.parquet: text_a, text_b, y; validation
entities excluded), under a wall-clock budget; then scores the uncertain validation and test pairs
(unc_val_text.parquet, unc_test_text.parquet: q, s1, prob, text_a, text_b) -> llm_val.parquet, llm_test.parquet (q, s1, logit).

python lora_judge.py  (env: TRAIN_MIN minutes of training, default 70; MODEL; SEED; MAXLEN; BS; CKPT_EVERY steps;
                      SCORE_ONLY=1 scores with the adapter saved in DATA/ckpt, e.g. after an interruption)
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B")
TRAIN_S = float(os.environ.get("TRAIN_MIN", "70")) * 60
MAXLEN, BS, LR = int(os.environ.get("MAXLEN", "128")), int(os.environ.get("BS", "32")), 1e-4
SEED = int(os.environ.get("SEED", "11"))
D = Path(os.environ.get("DATA", "."))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def fmt(a, b):
    return [f"Record 1: {x}\nRecord 2: {y}\nSame business?" for x, y in zip(a, b)]


torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
CKPT = D / "ckpt"
SCORE_ONLY = os.environ.get("SCORE_ONLY") == "1"   # skip training, score with the saved adapter in DATA/ckpt
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1000"))
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
if SCORE_ONLY:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(CKPT))
else:
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.cuda()
log("model loaded", MODEL, "(score only)" if SCORE_ONLY else "")
if not SCORE_ONLY:
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

tr = pl.read_parquet(D / "ce_train.parquet").sample(fraction=1.0, shuffle=True, seed=SEED)
if SCORE_ONLY:
    tr = tr.head(0)
a, b, y = tr["text_a"].to_list(), tr["text_b"].to_list(), tr["y"].to_numpy().astype(np.float32)
# length-bucketed batches: chunks of 50 batches sorted by length, then shuffled
lens = np.array([len(x) + len(z) for x, z in zip(a, b)])
order = []
rng = np.random.default_rng(SEED)
for i in range(0, len(a), BS * 50):
    idx = np.arange(i, min(i + BS * 50, len(a)))
    idx = idx[np.argsort(lens[idx])]
    bl = [idx[j:j + BS] for j in range(0, len(idx), BS)]
    rng.shuffle(bl)
    order.extend(bl)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
T_TRAIN0 = time.time()


def set_lr(step):  # warm-up 100 steps, then linear decay by wall-clock time to 5% at the budget end
    frac = min(1.0, (time.time() - T_TRAIN0) / max(TRAIN_S - (T_TRAIN0 - T0), 60))
    lr = LR * min(1.0, (step + 1) / 100) * max(0.05, 1.0 - frac)
    for g in opt.param_groups:
        g["lr"] = lr
lossf = torch.nn.BCEWithLogitsLoss()
model.train()
step, run = 0, []
for bi in order:
    if time.time() - T0 > TRAIN_S:
        break
    enc = tok(fmt([a[i] for i in bi], [b[i] for i in bi]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**enc).logits.squeeze(-1).float()
    loss = lossf(out, torch.tensor(y[bi], device="cuda"))
    if not torch.isfinite(loss):
        log("non-finite loss, skipping batch")
        opt.zero_grad(); continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    set_lr(step); opt.step(); opt.zero_grad()
    run.append(loss.item()); step += 1
    if step % 200 == 0:
        log(f"step {step} pairs {step * BS} loss {np.mean(run[-200:]):.4f}")
    if step % CKPT_EVERY == 0:
        model.save_pretrained(str(CKPT)); log(f"checkpoint saved at step {step}")
if not SCORE_ONLY:
    model.save_pretrained(str(CKPT))
log(f"training done: {step} steps, {step * BS} pairs; adapter in {CKPT}")
model.eval()


@torch.no_grad()
def score(src, dst):
    d = pl.read_parquet(D / src)
    ta, tb = d["text_a"].to_list(), d["text_b"].to_list()
    ln = np.array([len(x) + len(z) for x, z in zip(ta, tb)])
    idx = np.argsort(ln)
    res = np.zeros(len(ta), dtype=np.float32)
    for i in range(0, len(idx), 128):
        bi = idx[i:i + 128]
        enc = tok(fmt([ta[j] for j in bi], [tb[j] for j in bi]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res[bi] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (i // 128) % 400 == 0:
            log(f"score {src}: {i}/{len(idx)}")
    d.select("q", "s1").with_columns(pl.Series("logit", res)).write_parquet(D / dst)
    if "y" in d.columns:
        yy = d["y"].to_numpy()
        p = 1 / (1 + np.exp(-res))
        ll = -np.mean(yy * np.log(np.clip(p, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-6, 1)))
        log(f"{src}: logloss {ll:.4f}")
    log("wrote", dst)


score("unc_val_text.parquet", "llm_val.parquet")
score("unc_test_text.parquet", "llm_test.parquet")
log("DONE")
