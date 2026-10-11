"""Pre-register a matched-update experiment matrix. Dry-run unless --execute."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from abacus.config import preset
from abacus.training import select_device, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/add20")
    p.add_argument("--output", default="runs/week1")
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--continue", dest="continue_runs", action="store_true")
    p.add_argument("--only", nargs="+", help="subset of names shown by dry run")
    args = p.parse_args()
    pairs = [(n, 42) for n in ("nope", "fire", "abacus_fixed", "abacus", "abacus_ii", "abacus_loop")]
    pairs += [(n, 43) for n in ("nope", "abacus")]
    jobs = []
    for name, seed in pairs:
        label = f"{name}_seed{seed}"
        cfg = replace(preset(name, seed), steps=args.steps, micro_batch=args.micro_batch,
                      accumulation=args.accumulation).validate()
        jobs.append((label, cfg))
    if args.only and not set(args.only).issubset({j[0] for j in jobs}):
        raise ValueError("Unknown --only experiment name")
    plan = {"scope": "scaled addition reproduction; not original-paper accuracy",
            "jobs": [{"run": str(Path(args.output) / n), "config": c.dict()} for n, c in jobs]}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("DRY RUN: no training started. Add --execute after benchmark and protocol freeze.")
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        # For subsets, verify against the already frozen full plan rather than replacing it.
        old = json.loads(plan_path.read_text())
        old_jobs = {j["run"]: j["config"] for j in old["jobs"]}
        for job in plan["jobs"]:
            if old_jobs.get(job["run"]) != job["config"]:
                raise ValueError("Frozen plan mismatch. Use a new suite directory.")
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    device = select_device(args.device)
    for label, cfg in jobs:
        if args.only and label not in args.only:
            continue
        run = root / label
        status_path = run / "status.json"
        if args.continue_runs and status_path.exists() and json.loads(status_path.read_text())["completed"]:
            print(f"skip completed: {label}")
            continue
        last = run / "last.pt"
        resume = last if args.continue_runs and last.exists() else None
        train(cfg, args.data, run, device, resume)


if __name__ == "__main__":
    main()
