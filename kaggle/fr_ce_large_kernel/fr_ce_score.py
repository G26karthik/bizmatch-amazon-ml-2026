"""Kaggle GPU job: score France test pairs (original and France-cleaned texts) with the trained bge-reranker-v2-m3
cross-encoder. Both versions are scored in the same environment, so only the cleaning effect is transferred.
Inputs: kernel output of aml2026-cross-encoder-large (ce_model/), dataset aml2026-fr-ce (fr_ce_texts.parquet,
fr_ce_texts_orig.parquet). Outputs: fr_new.parquet, fr_orig.parquet (q, s1, pce)."""
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
IN = next(Path("/kaggle/input").rglob("fr_ce_texts.parquet")).parent
OUT = Path("/kaggle/working")
T0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda().eval()
net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
print(f"model {MODEL}, gpus {torch.cuda.device_count()}", flush=True)


@torch.inference_mode()
def score(src, dst):
    d = pl.read_parquet(IN / src)
    a, b = d["text_a"].to_list(), d["text_b"].to_list()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    p = np.empty(len(a), np.float32)
    for n, i in enumerate(range(0, len(order), 512)):
        j = order[i:i + 512]
        enc = tok([a[k] for k in j], [b[k] for k in j], truncation="longest_first", max_length=128, padding=True,
                  return_tensors="pt")
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p[j] = torch.sigmoid(net(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if n % 500 == 0:
            print(f"[{time.time() - T0:7.0f}s] {src}: {min(i + 512, len(a))}/{len(a)}", flush=True)
    d.select("q", "s1").with_columns(pl.Series("pce", p)).write_parquet(OUT / dst)


score("fr_ce_texts.parquet", "fr_new.parquet")
score("fr_ce_texts_orig.parquet", "fr_orig.parquet")
print("done", flush=True)
