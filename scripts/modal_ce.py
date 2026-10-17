"""Modal GPU job: train a cross-encoder on all 1.2M uncertain training pairs and score validation, test and the
France-cleaned test pairs. Same recipe family as the Kaggle kernels (kaggle/ce*_kernel), on one H100 instead of 2x T4.

Volume aml2026-ce holds the inputs (upload once):
    modal volume put aml2026-ce kaggle/ce3_ds/ce_train.parquet /data/ce_train.parquet   (also ce_val, ce_test)
    modal volume put aml2026-ce kaggle/fr_ce_ds/fr_ce_texts.parquet /data/fr_ce_texts.parquet
Outputs go to /out/<run>/ on the volume: ce_scores_val.parquet, ce_scores_test.parquet (default zone), fr_new.parquet
(France pairs with cleaned text; merge with scripts/france_ce_merge.py using ce_scores_test.parquet as --orig),
ce_model/ and train.log. A checkpoint is written every 20 minutes, so a preempted run resumes where it stopped.

modal run --detach scripts/modal_ce.py --runs bgefull,e5lfull
modal volume get aml2026-ce /out/bgefull local_data/modal/
"""
import modal

app = modal.App("aml2026-ce")
vol = modal.Volume.from_name("aml2026-ce", create_if_missing=True)
hf = modal.Volume.from_name("aml2026-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         # torch 2.11 + transformers 5.17 diverged after warm-up on both models (bf16: NaN; fp16: loss back to ln 2),
         # while the same recipe trained fine on Kaggle: use an older, well-tested stack
         .uv_pip_install("torch==2.6.0", "transformers==4.51.3", "polars==1.44.2", "numpy==2.2.6", "sentencepiece", "protobuf")
         .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "true"}))

# run name -> (base model, epochs, learning rate, train batch, seed); word embeddings frozen as in the Kaggle kernels
RUNS = {
    "bgefull": ("BAAI/bge-reranker-v2-m3", 2, 1e-5, 64, 5),         # Apache-2.0, 568M; Kaggle run used 700k pairs, 1 pass
    "e5lfull": ("intfloat/multilingual-e5-large", 2, 1e-5, 64, 42),  # MIT, 560M; Kaggle run: 700k pairs within 3.4 h
    "e5bfull3": ("intfloat/multilingual-e5-base", 3, 3e-5, 128, 23),  # MIT, 278M; best Kaggle recipe (2 passes) + 1 pass
    # Apache-2.0, 0.6B decoder (architecture diversity): one text per pair, last-token score head, bf16
    "qwen3r": ("Qwen/Qwen3-Reranker-0.6B", 2, 1e-5, 64, 13, {"concat": True, "bf16": True}),
}


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol, "/hf": hf}, timeout=3 * 3600)
def smoke() -> str:
    import torch
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    return f"{torch.__version__} {torch.cuda.get_device_name(0)} {float((x @ x).float().abs().mean()):.2f}"


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol, "/hf": hf}, timeout=3 * 3600)
def train_score(run: str, max_train_minutes: float = 110.0, maxlen: int = 128, bs_eval: int = 1024) -> str:
    import math
    import queue
    import threading
    import time
    from pathlib import Path

    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    base, epochs, lr, bs, seed, *extra = RUNS[run]
    cfg = extra[0] if extra else {}
    concat, bf16 = cfg.get("concat", False), cfg.get("bf16", False)
    amp = torch.bfloat16 if bf16 else torch.float16
    data, out = Path("/vol/data"), Path("/vol/out") / run
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / "train.log", "a")

    def log(*a):
        s = f"[{time.time() - t0:7.0f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    if concat:
        tok.padding_side = "right"

    def encode(a, b):
        if concat:  # decoder model: one text per pair
            t = [f"Business record 1: {x}\nBusiness record 2: {z}" for x, z in zip(a, b)]
            return dict(tok(t, truncation=True, max_length=maxlen + 32, padding=True, return_tensors="pt"))
        return dict(tok(a, b, truncation="longest_first", max_length=maxlen, padding=True, return_tensors="pt"))

    def prefetch(batch_iter, depth=48):  # tokenize on a background thread (the Rust tokenizer releases the GIL)
        q = queue.Queue(depth)

        def work():
            for item in batch_iter:
                q.put(item)
            q.put(None)
        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield item

    model_dir = out / "ce_model"
    if (model_dir / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float32).cuda()
        if concat:
            model.config.pad_token_id = tok.pad_token_id
        log("loaded trained model", model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, torch_dtype=torch.float32).cuda()
        model.get_input_embeddings().weight.requires_grad_(False)
        if concat:
            model.config.pad_token_id = tok.pad_token_id
        tr = pl.read_parquet(data / "ce_train.parquet")
        ta, tb = tr["text_a"].to_list(), tr["text_b"].to_list()
        y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
        # length-bucketed order as in the Kaggle kernels: random chunks of 64 batches sorted by length, batches shuffled
        rng = np.random.default_rng(seed)
        lens = np.array([len(x) + len(z) for x, z in zip(ta, tb)])
        perm = rng.permutation(len(ta))
        chunk, bl = bs * 64, []
        for i in range(0, len(perm), chunk):
            c = perm[i:i + chunk]
            c = c[np.argsort(lens[c], kind="stable")]
            bl += [c[k:k + bs] for k in range(0, len(c), bs)]
        batches = [bl[i] for _ in range(epochs) for i in rng.permutation(len(bl))]
        total = len(batches)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, 1000, total)
        start, ck, st = 0, out / "ckpt.pt", {}
        if ck.exists():
            st = torch.load(ck, map_location="cuda", weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sch.load_state_dict(st["sch"])
            start = st["step"]
            log(f"resumed from checkpoint at step {start}")
        # fp16 autocast with a gradient scaler, as in the Kaggle kernels: the scaler skips steps whose gradients overflow
        # (plain bf16 without it went to NaN within 1000 steps on both models)
        scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
        if ck.exists() and "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        bad = 0
        log(f"{run}: {base}, train pairs {len(ta)}, positives {float(y.mean()):.3f}, steps {total}, "
            f"trainable {sum(p.numel() for p in params) / 1e6:.0f}M")
        t_train, t_ck = time.time(), time.time()
        it = prefetch((s, j, encode([ta[k] for k in j], [tb[k] for k in j])) for s, j in enumerate(batches) if s >= start)
        for s, j, enc in it:
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=amp):
                logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), y[j].cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if s % 200 == 0:
                lv = loss.item()
                bad = bad + 1 if not math.isfinite(lv) else 0
                if bad >= 5:
                    raise RuntimeError(f"loss not finite at step {s}: stopping instead of burning GPU time")
            if s % 1000 == 0 or (s < 3000 and s % 200 == 0):
                el = time.time() - t_train
                log(f"step {s}/{total} loss {loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} "
                    f"{(s - start + 1) / max(el, 1e-9):.1f} steps/s")
            if time.time() - t_ck > 20 * 60:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                            "scaler": scaler.state_dict(), "step": s + 1}, ck)
                vol.commit()
                t_ck = time.time()
                log(f"checkpoint at step {s + 1}")
            if time.time() - t_train > max_train_minutes * 60:
                log(f"training budget reached at step {s}/{total}")
                break
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        if ck.exists():
            ck.unlink()
        vol.commit()
        log("saved model")
    model.eval()

    @torch.inference_mode()
    def score(path: Path, name: str):
        if (out / f"{name}.parquet").exists():
            log(f"{name} already scored")
            return
        d = pl.read_parquet(path)
        a, b = d["text_a"].to_list(), d["text_b"].to_list()
        order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # length-sorted batches run much faster
        p = np.empty(len(a), np.float32)
        it = prefetch((order[i:i + bs_eval], encode([a[k] for k in order[i:i + bs_eval]], [b[k] for k in order[i:i + bs_eval]]))
                      for i in range(0, len(order), bs_eval))
        for n, (j, enc) in enumerate(it):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=amp):
                p[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if n % 1000 == 0:
                log(f"score {name}: {min((n + 1) * bs_eval, len(a))}/{len(a)}")
        res = d.select("q", "s1").with_columns(pl.Series("pce", p))
        if "y" in d.columns:
            yy = d["y"].to_numpy()
            log(f"{name}: logloss {-np.mean(yy * np.log(p + 1e-7) + (1 - yy) * np.log(1 - p + 1e-7)):.4f}")
        res.write_parquet(out / f"{name}.parquet")
        vol.commit()
        log(f"scored {name}: {len(a)}")

    score(data / "ce_val.parquet", "ce_scores_val")
    score(data / "ce_test.parquet", "ce_scores_test")
    score(data / "fr_ce_texts.parquet", "fr_new")
    log("done")
    return f"{run} done in {(time.time() - t0) / 60:.0f} min"


@app.local_entrypoint()
def main(runs: str = "bgefull", minutes: float = 110.0, smoke_only: bool = False):
    if smoke_only:
        print(smoke.remote())
        return
    calls = [train_score.spawn(r, minutes) for r in runs.split(",")]
    for c in calls:
        print(c.get())
