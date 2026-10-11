"""All commands work from the project directory on Windows, Linux or WSL2."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from abacus.config import PRESETS, preset
from abacus.data import prepare, read_jsonl, sha256, split_of, example
from abacus.evaluation import evaluate, generate
from abacus.training import benchmark, load_model, select_device, train


def add_model_options(parser):
    parser.add_argument("--preset", choices=PRESETS, default="abacus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--tiny", action="store_true", help="CPU correctness test only; NOT research scale")


def config(args):
    c = replace(preset(args.preset, args.seed), steps=args.steps,
                micro_batch=args.micro_batch, accumulation=args.accumulation,
                precision=args.precision)
    if args.tiny:
        c = replace(c, d_model=32, d_ff=64, heads=4, block_layers=2,
                    train_max_digits=2, warmup=2, eval_every=5, save_every=5)
    return c.validate()


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        if sha256(root / filename) != digest:
            raise ValueError(f"Checksum failed: {filename}")
    for split in ("val", "test"):
        seen = set()
        for row in read_jsonl(root / f"{split}.jsonl"):
            a, b = int(row["a"]), int(row["b"])
            assert split_of(a, b) == split, row
            assert example(a, b) == row
            assert (a, b) not in seen
            seen.add((a, b))
    # Full training audit scans all rows, not only a small sample.
    import numpy as np
    from abacus.data import PAD, BOS, EOS, decode
    data = np.load(root / "train.npy", mmap_mode="r")
    counts = np.zeros((manifest["train_max"], manifest["train_max"]), dtype=np.int64)
    for row in data:
        valid = row[row != PAD].tolist()
        assert valid[0] == BOS and valid[-1] == EOS
        question, target = decode(valid[1:-1]).split("=")
        aa, bb = question.split("+")
        a, b = int(aa[::-1]), int(bb[::-1])
        assert str(a)[::-1] == aa and str(b)[::-1] == bb
        assert target == str(a + b)[::-1]
        assert split_of(a, b) == "train"
        counts[len(aa) - 1, len(bb) - 1] += 1
    assert int(counts.min()) == int(counts.max())
    result = {"status": "pass", "train_rows_checked": len(data),
              "equal_rows_per_length_cell": int(counts.min()), "checksums": "pass",
              "arithmetic": "pass", "canonical_split_isolation": "pass"}
    (root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description="4060 resource-scaled Abacus reproduction")
    commands = p.add_subparsers(dest="command", required=True)
    s = commands.add_parser("prepare")
    s.add_argument("--output", default="data/add20")
    s.add_argument("--samples", type=int, default=200_000)
    s.add_argument("--train-max", type=int, default=20)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--per-cell", type=int, default=64)
    s.add_argument("--test-lengths", type=int, nargs="+")
    s = commands.add_parser("audit")
    s.add_argument("--data", default="data/add20")
    s = commands.add_parser("benchmark")
    add_model_options(s)
    s.add_argument("--data", default=None)
    s.add_argument("--updates", type=int, default=30)
    s.add_argument("--output", default="benchmark.json")
    s = commands.add_parser("train")
    add_model_options(s)
    s.add_argument("--data", default="data/add20")
    s.add_argument("--run", required=True)
    s.add_argument("--resume")
    s.add_argument("--max-hours", type=float, default=None, help="session limit; does not alter planned schedule")
    s = commands.add_parser("eval")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--examples", default="data/add20/test.jsonl")
    s.add_argument("--output", required=True)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    s.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    s = commands.add_parser("demo")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--a", type=int, default=123)
    s.add_argument("--b", type=int, default=45)
    s.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.samples, args.train_max, args.seed,
                         args.per_cell, args.test_lengths)
    elif args.command == "audit":
        result = audit(args.data)
    elif args.command == "benchmark":
        result = benchmark(config(args), select_device(args.device), args.data, args.updates)
        out = Path(args.output)
        if out.exists():
            raise FileExistsError("Use a fresh benchmark output path")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "train":
        result = train(config(args), args.data, args.run, select_device(args.device),
                       args.resume, args.max_hours)
    elif args.command == "eval":
        device = select_device(args.device)
        model, saved = load_model(args.checkpoint, device)
        result, _ = evaluate(model, read_jsonl(args.examples), device, args.batch, args.precision, args.output)
        status_file = Path(args.checkpoint).parent / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        completed = status.get("completed", saved["step"] == model.cfg.steps)
        provenance = {"checkpoint": Path(args.checkpoint).as_posix(),
                      "checkpoint_sha256": sha256(args.checkpoint), "step": saved["step"],
                      "planned_steps": model.cfg.steps, "incomplete_run": not completed,
                      "completed_steps": status.get("step", saved["step"]),
                      "data_manifest_sha256": saved["manifest_hash"],
                      "examples_sha256": sha256(args.examples), "config": model.cfg.dict()}
        Path(args.output, "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    else:
        if args.a < 0 or args.b < 0:
            raise ValueError("This reproduction trains nonnegative-format positive integers, not signed addition")
        device = select_device(args.device)
        model, _ = load_model(args.checkpoint, device)
        prompt = f"{str(args.a)[::-1]}+{str(args.b)[::-1]}="
        pred = generate(model, [prompt], max(len(str(args.a)), len(str(args.b))) + 2, device)[0]
        # Python ground truth is used ONLY AFTER network prediction, as an evaluator.
        result = {"prompt": prompt, "network": pred,
                  "normal_order_prediction": pred["prediction"][::-1],
                  "ground_truth_after_decoding": args.a + args.b,
                  "correct": pred["ended"] and pred["prediction"] == str(args.a + args.b)[::-1]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
