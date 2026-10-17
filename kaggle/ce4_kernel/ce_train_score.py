"""Kaggle GPU job: large cross-encoder for uncertain candidate pairs.

Base model BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M, XLM-R large reranker) with its 1-logit head, fine-tuned on
training pairs from the uncertain zone (validation entities excluded), then used to score validation and test pairs.
Inputs: dataset aml2026-ce-data-big (ce_train.parquet (text_a, text_b, y), ce_val.parquet, ce_test.parquet) and
dataset aml2026-ce-extra (wider-zone ce_val.parquet, ce_test.parquet).
Outputs: ce_scores_val.parquet, ce_scores_test.parquet (default zone), ce_extra_scores_val.parquet (wider zone),
ce_model/ (weights). Wider-zone test pairs are scored by a separate job (time limit).
Speed: word embeddings frozen, length-bucketed training batches, training stopped at a wall-clock budget.
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
EXTRA = next(p.parent for p in Path("/kaggle/input").rglob("ce_val.parquet") if p.parent != IN)
OUT = Path("/kaggle/working")
BASE = "BAAI/bge-reranker-v2-m3"
MAXLEN, BS_TRAIN, BS_EVAL, LR, SEED = 128, 64, 512, 1e-5, 42
N_TRAIN, TRAIN_BUDGET = 700_000, 3.4 * 3600  # pairs; seconds of training before moving on to scoring
T0 = time.time()
torch.manual_seed(SEED)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
ngpu = torch.cuda.device_count()
net = torch.nn.DataParallel(model) if ngpu > 1 else model
log(f"gpus {ngpu}, trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.0f}M")


def batches(a, b, bs, order):
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok([a[k] for k in j], [b[k] for k in j], truncation="longest_first", max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        yield j, {k: v.cuda(non_blocking=True) for k, v in enc.items()}


tr = pl.read_parquet(IN / "ce_train.parquet").sample(n=N_TRAIN, seed=SEED, shuffle=True)
a, b, y = tr["text_a"].to_list(), tr["text_b"].to_list(), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
log(f"train pairs {len(a)}, positives {float(y.mean()):.3f}")
# length-bucketed order: random chunks of 64 batches, sorted by length inside each chunk, chunk batches shuffled
rng = np.random.default_rng(SEED)
lens = np.array([len(x) + len(z) for x, z in zip(a, b)])
perm = rng.permutation(len(a))
chunk = BS_TRAIN * 64
bl = []
for i in range(0, len(perm), chunk):
    c = perm[i:i + chunk]
    c = c[np.argsort(lens[c], kind="stable")]
    bl += [c[k:k + BS_TRAIN] for k in range(0, len(c), BS_TRAIN)]
bl = [bl[i] for i in rng.permutation(len(bl))]
order = np.concatenate(bl)
steps = math.ceil(len(a) / BS_TRAIN)
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, 1000, steps)
scaler = torch.amp.GradScaler("cuda")
lossf = torch.nn.BCEWithLogitsLoss()
net.train()
t_train = time.time()
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
    if time.time() - t_train > TRAIN_BUDGET:
        log(f"training budget reached at step {s}/{steps}")
        break
model.save_pretrained(OUT / "ce_model")
tok.save_pretrained(OUT / "ce_model")
net.eval()


@torch.inference_mode()
def score(src: Path, name: str, out_name: str):
    d = pl.read_parquet(src / f"ce_{name}.parquet")
    a, b = d["text_a"].to_list(), d["text_b"].to_list()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # length-sorted batches run much faster
    p = np.empty(len(a), np.float32)
    for n, (j, enc) in enumerate(batches(a, b, BS_EVAL, order)):
        with torch.autocast("cuda", dtype=torch.float16):
            p[j] = torch.sigmoid(net(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if n % 500 == 0:
            log(f"score {out_name}: {min((n + 1) * BS_EVAL, len(a))}/{len(a)}")
    out = d.select("q", "s1").with_columns(pl.Series("pce", p))
    if "y" in d.columns:
        yy = d["y"].to_numpy()
        log(f"{out_name}: logloss {-np.mean(yy * np.log(p + 1e-7) + (1 - yy) * np.log(1 - p + 1e-7)):.4f}")
    out.write_parquet(OUT / f"{out_name}.parquet")


score(IN, "val", "ce_scores_val")
score(EXTRA, "val", "ce_extra_scores_val")
score(IN, "test", "ce_scores_test")
log("done")
