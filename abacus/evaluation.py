import contextlib
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .data import BOS, EOS, PAD, DIGIT_MIN, DIGIT_MAX, decode, encode
from .model import local_positions


def amp_context(device, precision="auto"):
    if str(device).startswith("cuda"):
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Explicit BF16 requested but unsupported; choose auto or fp16")
        dtype = torch.bfloat16 if precision in {"auto", "bf16"} and torch.cuda.is_bf16_supported() else torch.float16
        if precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.inference_mode()
def generate(model, prompts, max_new_tokens, device, precision="auto", use_cache=True):
    """Only prompts are inputs. No ground truth or oracle carries enter inference.

    Group callers by equal prompt length. Do not pad BOS/inputs to 'align' digits.
    No vocabulary constraints: a predicted '+' is a genuine error, not repaired.
    """
    if not prompts:
        return []
    lengths = {len(p) for p in prompts}
    if len(lengths) != 1:
        raise ValueError("Group equal-length prompts before decoding")
    model.eval()
    full = torch.tensor([[BOS] + encode(p) for p in prompts], device=device)
    if full.size(1) + max_new_tokens - 1 > model.cfg.max_positions:
        raise ValueError("Requested decoding context exceeds model maximum")
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    generated = []
    cache, inp = None, full
    pos = local_positions(inp)
    was_digit = ((inp[:, -1] >= DIGIT_MIN) & (inp[:, -1] <= DIGIT_MAX))
    last_local = pos[:, -1]
    with amp_context(device, precision):
        for step in range(max_new_tokens):
            logits, cache = model(inp, positions=pos, cache=cache if use_cache else None,
                                  use_cache=use_cache)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            generated.append(nxt)
            done = done | (nxt == EOS)
            if bool(done.all()):
                break
            full = torch.cat((full, nxt[:, None]), 1)
            if use_cache:
                digit = (nxt >= DIGIT_MIN) & (nxt <= DIGIT_MAX)
                last_local = torch.where(digit, torch.where(was_digit, last_local + 1,
                                                           torch.ones_like(last_local)),
                                         torch.zeros_like(last_local))
                was_digit = digit
                inp, pos = nxt[:, None], last_local[:, None]
            else:
                inp, pos = full, local_positions(full)
    all_ids = torch.stack(generated, 1).cpu().tolist()
    out = []
    for row in all_ids:
        ended = EOS in row
        prefix = row[:row.index(EOS)] if ended else row
        # A missing EOS is an error even when the digit prefix is correct.
        out.append({"prediction": decode(prefix), "ended": ended})
    return out


def wilson(success, total, z=1.96):
    if not total:
        return [None, None]
    p = success / total
    denom = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0, mid - half), min(1, mid + half)]


def carry_profile(a, b):
    aa, bb = str(a)[::-1], str(b)[::-1]
    carry = longest = run = count = 0
    for idx in range(max(len(aa), len(bb))):
        s = (int(aa[idx]) if idx < len(aa) else 0) + (int(bb[idx]) if idx < len(bb) else 0) + carry
        carry = int(s >= 10)
        count += carry
        run = run + 1 if carry else 0
        longest = max(longest, run)
    return {"carry_count": count, "longest_carry_chain": longest}


def evaluate(model, examples, device, batch_size=32, precision="auto", output=None):
    cells = defaultdict(list)
    for ex in examples:
        cells[(ex["i"], ex["j"])].append(ex)
    grid, errors, result_rows = [], [], []
    for (i, j), rows in sorted(cells.items()):
        good, eos_good = 0, 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            outputs = generate(model, [e["prompt"] for e in chunk], max(i, j) + 2,
                               device, precision)
            for ex, pred in zip(chunk, outputs):
                ok = pred["ended"] and pred["prediction"] == ex["answer"]
                good += int(ok)
                eos_good += int(pred["ended"])
                row = dict(ex, **pred, correct=ok, **carry_profile(ex["a"], ex["b"]))
                result_rows.append(row)
                if not ok:
                    errors.append(row)
        grid.append({"i": i, "j": j, "n": len(rows), "correct": good,
                     "accuracy": good / len(rows), "eos_rate": eos_good / len(rows),
                     "wilson95_low": wilson(good, len(rows))[0],
                     "wilson95_high": wilson(good, len(rows))[1]})
    L = model.cfg.train_max_digits
    summary = {"evaluation": "greedy autoregressive exact match incl. EOS",
               "train_max_digits": L, "grid_is_sparse": True,
               "group_definitions": {"ID": "both lengths <= train_max",
                                     "OOD": "at least one length > train_max, both <=100"}}
    for name, select in [("ID", lambda g: max(g["i"], g["j"]) <= L),
                         ("OOD", lambda g: L < max(g["i"], g["j"]) <= 100)]:
        selected = [g for g in grid if select(g)]
        n = sum(g["n"] for g in selected)
        good = sum(g["correct"] for g in selected)
        summary[name] = {"n": n, "correct": good, "micro_accuracy": good / n if n else None,
                         "macro_cell_accuracy": sum(g["accuracy"] for g in selected) / len(selected) if selected else None,
                         "micro_wilson95": wilson(good, n), "cells": len(selected)}
    if output:
        output = Path(output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
        output.mkdir(parents=True)
        with open(output / "grid.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(grid[0]))
            w.writeheader()
            w.writerows(grid)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for filename, records in [("predictions.jsonl", result_rows), ("errors.jsonl", errors)]:
            with open(output / filename, "w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row) + "\n")
    return summary, grid
