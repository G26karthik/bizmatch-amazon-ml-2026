"""Kaggle GPU job: score the wider-zone test pairs with the trained bge-reranker-v2-m3 cross-encoder (no training).
Its wider-zone validation pairs were already scored by the training job (ce_extra_scores_val.parquet).

Inputs: kernel output of aml2026-cross-encoder-large (ce_model/), dataset aml2026-ce-extra (ce_test.parquet).
Output: ce_extra_scores_test.parquet (q, s1, pce).
"""
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

MODEL = next(Path("/kaggle/input").rglob("ce_model/config.json")).parent
IN = next(Path("/kaggle/input").rglob("ce_test.parquet")).parent
OUT = Path("/kaggle/working")
MAXLEN, BS_EVAL = 128, 512
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda().eval()
net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
log(f"model {MODEL}, gpus {torch.cuda.device_count()}")


@torch.inference_mode()
def score(name):
    d = pl.read_parquet(IN / f"ce_{name}.parquet")
    a, b = d["text_a"].to_list(), d["text_b"].to_list()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    p = np.empty(len(a), np.float32)
    for n, i in enumerate(range(0, len(order), BS_EVAL)):
        j = order[i:i + BS_EVAL]
        enc = tok([a[k] for k in j], [b[k] for k in j], truncation="longest_first", max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p[j] = torch.sigmoid(net(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if n % 500 == 0:
            log(f"{name}: {min(i + BS_EVAL, len(a))}/{len(a)}")
    d.select("q", "s1").with_columns(pl.Series("pce", p)).write_parquet(OUT / f"ce_extra_scores_{name}.parquet")


score("test")
log("done")
