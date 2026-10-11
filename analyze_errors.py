"""Summarize actual predictions; carry statistics are diagnostics, not model inputs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    groups, kinds = defaultdict(lambda: [0, 0]), Counter()
    train_max = json.loads((Path(a.eval) / "summary.json").read_text())["train_max_digits"]
    rows = [json.loads(s) for s in (Path(a.eval) / "predictions.jsonl").read_text().splitlines() if s]
    for r in rows:
        chain = r["longest_carry_chain"]
        bucket = "0" if chain == 0 else "1-2" if chain <= 2 else "3-5" if chain <= 5 else "6+"
        label = ("ID" if max(r["i"], r["j"]) <= train_max else "OOD") + ":carry_chain=" + bucket
        groups[label][0] += 1
        groups[label][1] += int(r["correct"])
        if r["correct"]:
            continue
        pred = r["prediction"]
        if not r["ended"]:
            kinds["missing_EOS"] += 1
        elif not pred.isdigit():
            kinds["illegal_output_token"] += 1
        elif len(pred) != len(r["answer"]):
            kinds["wrong_digit_length"] += 1
        else:
            kinds["wrong_digit_value"] += 1
    out = {"mutually_exclusive_error_categories": dict(kinds),
           "carry_groups": {k: {"n": v[0], "correct": v[1], "accuracy": v[1] / v[0]} for k, v in sorted(groups.items())},
           "caution": "random test data: long carry-chain groups may have few samples; correlation is not causation"}
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError("Refusing to overwrite diagnostics")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
