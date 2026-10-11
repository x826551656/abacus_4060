import json
import math
import os
import platform
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import BatchStream, collate, read_jsonl, sha256
from .evaluation import amp_context, evaluate
from .model import ArithmeticTransformer, answer_loss


def select_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA torch and check driver; do not silently train on CPU.")
    return torch.device(request)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Repeatability on the same stack, not bitwise guarantees across GPUs/versions.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def environment(device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "vram_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None}


def lr_at(step, cfg):
    if step <= cfg.warmup:
        return cfg.lr * step / max(1, cfg.warmup)
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p)) / 2


def objective(model, x, y, beta, short_r):
    logits, _ = model(x, beta=beta)
    primary = answer_loss(logits, y)
    alpha = model.cfg.progressive_alpha
    if alpha:
        small_logits, _ = model(x, beta=beta, recurrences=short_r)
        return (1 - alpha) * primary + alpha * answer_loss(small_logits, y)
    return primary


def make_scaler(device, precision):
    enabled = device.type == "cuda" and precision != "fp32" and not (
        precision in {"auto", "bf16"} and torch.cuda.is_bf16_supported())
    return torch.amp.GradScaler("cuda", enabled=enabled)


def checkpoint(path, model, optimizer, scaler, stream, step, best, rng, manifest_hash):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "config": model.cfg.dict(), "step": step,
               "best_val": best, "stream": stream.state(), "method_rng": rng.getstate(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               "manifest_hash": manifest_hash}
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # Atomic replacement of THIS run's recoverable checkpoint.


def load_model(path, device):
    # Only load checkpoints you trust: torch pickle is not a safe interchange format.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**obj["config"]).validate()
    model = ArithmeticTransformer(cfg).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model, obj


def train(cfg, data_dir, run_dir, device, resume=None, max_hours=None):
    cfg.validate()
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["train_max"] != cfg.train_max_digits:
        raise ValueError("Training max digits and data manifest disagree")
    manifest_hash = sha256(data_dir / "manifest.json")
    for filename, digest in manifest["files"].items():
        if sha256(data_dir / filename) != digest:
            raise ValueError(f"Dataset changed after preparation: {filename}")
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError("Use a new run directory or --resume; refusing to overwrite a run")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    model = ArithmeticTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 betas=(0.9, 0.95), foreach=False)
    scaler = make_scaler(device, cfg.precision)
    rng = random.Random(cfg.seed + 719)
    start, best, state = 0, -1.0, None
    if resume:
        if not Path(resume).parent.samefile(run_dir):
            raise ValueError("Resume must use a checkpoint from the same run directory")
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["config"] != cfg.dict() or saved["manifest_hash"] != manifest_hash:
            raise ValueError("Resume requires IDENTICAL config and dataset; no silent LR-budget changes")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # PyTorch casts optimizer tensors to parameter devices, while preserving
        # Adam's non-capturable step counters on CPU where required.
        scaler.load_state_dict(saved["scaler"])
        start, best, state = saved["step"], saved["best_val"], saved["stream"]
        rng.setstate(saved["method_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    stream = BatchStream(data_dir, cfg.micro_batch, cfg.seed + 1234, state)
    val = read_jsonl(data_dir / "val.jsonl")
    meta = {"config": cfg.dict(), "environment": environment(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "data_manifest_sha256": manifest_hash, "data_manifest": manifest}
    if not resume:
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["environment"]), flush=True)
    print(f"parameters={meta['parameters']:,}, effective batch={cfg.micro_batch * cfg.accumulation}", flush=True)
    begin = time.perf_counter()
    log = open(run_dir / "train.jsonl", "a", encoding="utf-8")
    step = start
    try:
        for step in range(start + 1, cfg.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rate = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss_sum = 0.0
            batches = [collate(stream.next(), device) for _ in range(cfg.accumulation)]
            token_counts = [int((y != -100).sum()) for _, y in batches]
            total_tokens = sum(token_counts)
            for (x, y), count in zip(batches, token_counts):
                beta = rng.randint(1, cfg.offset_k) if model.abacus is not None else 1
                short_r = rng.randint(1, cfg.recurrences - 1) if cfg.progressive_alpha else None
                with amp_context(device, cfg.precision):
                    loss = objective(model, x, y, beta, short_r)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite loss. Keep existing checkpoint; reduce LR in a NEW run.")
                weight = count / total_tokens
                loss_sum += float(loss.detach()) * weight
                scaler.scale(loss * weight).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            record = {"step": step, "loss": loss_sum, "lr": rate,
                      "grad_norm": float(grad_norm), "elapsed_session_seconds": time.perf_counter() - begin,
                      "samples_seen": step * cfg.micro_batch * cfg.accumulation,
                      "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0}
            if step % 100 == 0 or step == 1:
                print(json.dumps(record), flush=True)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                summary, _ = evaluate(model, val, device, cfg.micro_batch, cfg.precision)
                score = summary["ID"]["macro_cell_accuracy"]
                record["val_ID_macro_exact_match"] = score
                print(f"validation (not test): {score:.4%}", flush=True)
                if score > best:
                    best = score
                    checkpoint(run_dir / "best.pt", model, optimizer, scaler, stream, step, best, rng, manifest_hash)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % cfg.save_every == 0 or step == cfg.steps:
                checkpoint(run_dir / "last.pt", model, optimizer, scaler, stream, step, best, rng, manifest_hash)
            if max_hours is not None and time.perf_counter() - begin >= max_hours * 3600:
                checkpoint(run_dir / "last.pt", model, optimizer, scaler, stream, step, best, rng, manifest_hash)
                print("Session time budget reached; run is INCOMPLETE. Resume with unchanged --steps.", flush=True)
                break
    except KeyboardInterrupt:
        print("Interrupted: use the last completed checkpoint. Partial accumulation is deliberately NOT saved.", flush=True)
        raise
    finally:
        log.close()
    result = {"completed": step == cfg.steps, "step": step, "planned_steps": cfg.steps,
              "best_val_ID_macro": best, "session_seconds": time.perf_counter() - begin}
    (run_dir / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def benchmark(cfg, device, data_dir=None, updates=30, warmup_updates=5):
    """Measures real optimizer updates INCLUDING accumulation/progressive passes."""
    seed_all(cfg.seed)
    model = ArithmeticTransformer(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, foreach=False)
    scaler = make_scaler(device, cfg.precision)
    stream = BatchStream(data_dir, cfg.micro_batch, cfg.seed) if data_dir else None
    rng = random.Random(cfg.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for update in range(updates + warmup_updates):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batches = []
        for _ in range(cfg.accumulation):
            if stream:
                x, y = collate(stream.next(), device)
            else:
                # Conservative maximum train length; real dataset benchmarks are preferred.
                t = 3 * cfg.train_max_digits + 4
                x = torch.randint(3, 13, (cfg.micro_batch, t), device=device)
                x[:, [0, cfg.train_max_digits + 1, 2 * cfg.train_max_digits + 2]] = 13
                y = torch.randint(3, 13, x.shape, device=device)
                y[:, :2 * cfg.train_max_digits + 2] = -100
            batches.append((x, y))
        token_counts = [int((y != -100).sum()) for _, y in batches]
        total_tokens = sum(token_counts)
        for (x, y), count in zip(batches, token_counts):
            beta = rng.randint(1, cfg.offset_k) if model.abacus is not None else 1
            short_r = 1 if cfg.progressive_alpha else None
            with amp_context(device, cfg.precision):
                loss = objective(model, x, y, beta, short_r) * (count / total_tokens)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if update >= warmup_updates:
            timings.append(time.perf_counter() - tick)
    sec = float(np.median(timings))
    return {"config": cfg.dict(), "environment": environment(device),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "median_seconds_per_update": sec,
            "projected_train_hours_excluding_eval": sec * cfg.steps / 3600,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9 if device.type == "cuda" else None,
            "measured_updates": updates, "warmup_updates": warmup_updates}
