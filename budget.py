"""Conservative week budget from YOUR measurements, not invented GPU throughput."""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", nargs="+", required=True)
    p.add_argument("--gpu-hours", type=float, default=100)
    p.add_argument("--reserve-hours", type=float, default=20)
    p.add_argument("--ceiling-steps", type=int, default=24000)
    p.add_argument("--output", default="budget.json")
    a = p.parse_args()
    rows = [json.loads(Path(f).read_text()) for f in a.benchmarks]
    names = {r["config"]["name"] for r in rows}
    needed = {"nope", "fire", "abacus_fixed", "abacus", "abacus_ii", "abacus_loop"}
    if names != needed or len(rows) != 6:
        raise ValueError("Provide exactly one real benchmark per six core presets")
    if any(r["environment"]["gpu"] is None for r in rows):
        raise ValueError("CPU measurements cannot budget a 4060 GPU week")
    fields = ("d_model", "heads", "d_ff", "micro_batch", "accumulation", "precision", "train_max_digits")
    if len({tuple(r["config"][k] for k in fields) for r in rows}) != 1:
        raise ValueError("Benchmark configs must use the same width and effective batch settings")
    if len({json.dumps(r["environment"], sort_keys=True) for r in rows}) != 1:
        raise ValueError("Measure all presets on the same GPU/software stack")
    available = a.gpu_hours - a.reserve_hours
    if available <= 0:
        raise ValueError("GPU hours must exceed reserved hours")
    # One seed for all six + a second seed for NoPE and Abacus = eight runs.
    seconds_per_round = sum(r["median_seconds_per_update"] * (2 if r["config"]["name"] in {"nope", "abacus"} else 1)
                            for r in rows)
    # 25% training overhead margin; evaluation remains separately reserved.
    limit = math.floor(available * 3600 / (seconds_per_round * 1.25) / 1000) * 1000
    steps = min(a.ceiling_steps, limit)
    result = {"gpu_hours_available": a.gpu_hours, "reserve_for_evaluation_and_failures_hours": a.reserve_hours,
              "matched_steps_per_run": steps, "runs": 8, "training_safety_multiplier": 1.25,
              "estimated_train_hours_with_margin": seconds_per_round * steps * 1.25 / 3600,
              "estimated_seen_examples_per_run": steps * rows[0]["config"]["micro_batch"] * rows[0]["config"]["accumulation"],
              "advice": "freeze suite with this step count BEFORE inspecting OOD test outcomes" if steps >= 4000 else
                        "insufficient budget for eight useful runs; prioritize NoPE/Abacus only or allow more GPU hours",
              "warning": "measurement-based estimate, NOT a completion or accuracy guarantee"}
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError("Use a new budget filename")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
