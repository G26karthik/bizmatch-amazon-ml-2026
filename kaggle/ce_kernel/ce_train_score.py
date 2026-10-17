"""Kaggle GPU job: cross-encoder for uncertain candidate pairs.

Base model intfloat/multilingual-e5-small (MIT, 118M) with a 1-logit classification head, fine-tuned on training pairs
from the uncertain zone (validation entities excluded), then used to score validation and test pairs.
Inputs (dataset aml2026-ce-data): ce_train.parquet (text_a, text_b, y), ce_val.parquet, ce_test.parquet (q, s1, text_a, text_b).
Outputs: ce_scores_val.parquet, ce_scores_test.parquet (q, s1, pce), ce_model/ (weights, for reproducibility).
"""
import math
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.optimization import get_linear_schedule_with_warmup  # noqa: E402

IN = next(Path("/kaggle/input").rglob("ce_train.parquet")).parent
OUT = Path("/kaggle/working")
BASE = "intfloat/multilingual-e5-small"
MAXLEN, BS_TRAIN, BS_EVAL, LR, SEED = 128, 128, 1024, 3e-5, 42
T0 = time.time()
torch.manual_seed(SEED)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
ngpu = torch.cuda.device_count()
net = torch.nn.DataParallel(model) if ngpu > 1 else model
log(f"gpus {ngpu}")


def batches(a, b, bs, order=None):
    idx = order if order is not None else np.arange(len(a))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([a[k] for k in j], [b[k] for k in j], truncation="longest_first", max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        yield j, {k: v.cuda(non_blocking=True) for k, v in enc.items()}


tr = pl.read_parquet(IN / "ce_train.parquet")
a, b, y = tr["text_a"].to_list(), tr["text_b"].to_list(), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
steps = math.ceil(len(a) / BS_TRAIN)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, 500, steps)
scaler = torch.amp.GradScaler("cuda")
lossf = torch.nn.BCEWithLogitsLoss()
net.train()
order = np.random.default_rng(SEED).permutation(len(a))
for s, (j, enc) in enumerate(batches(a, b, BS_TRAIN, order)):
    with torch.autocast("cuda", dtype=torch.float16):
        logit = net(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), y[j].cuda())
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sch.step()
    if s % 500 == 0:
        log(f"step {s}/{steps} loss {loss.item():.4f}")
model.save_pretrained(OUT / "ce_model")
tok.save_pretrained(OUT / "ce_model")
net.eval()


@torch.inference_mode()
def score(name):
    d = pl.read_parquet(IN / f"ce_{name}.parquet")
    a, b = d["text_a"].to_list(), d["text_b"].to_list()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # length-sorted batches run much faster
    p = np.empty(len(a), np.float32)
    for n, (j, enc) in enumerate(batches(a, b, BS_EVAL, order)):
        with torch.autocast("cuda", dtype=torch.float16):
            p[j] = torch.sigmoid(net(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if n % 200 == 0:
            log(f"score {name}: {min((n + 1) * BS_EVAL, len(a))}/{len(a)}")
    out = d.select("q", "s1").with_columns(pl.Series("pce", p))
    if "y" in d.columns:
        yy = d["y"].to_numpy()
        log(f"{name}: logloss {-np.mean(yy * np.log(p + 1e-7) + (1 - yy) * np.log(1 - p + 1e-7)):.4f}")
    out.write_parquet(OUT / f"ce_scores_{name}.parquet")


score("val")
score("test")
log("done")
