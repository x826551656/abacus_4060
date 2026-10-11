"""Plot REAL eval outputs only. Never generates a mock research heatmap."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", nargs="+", required=True)
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh report directory")
    out.mkdir(parents=True, exist_ok=True)
    all_summary, grouped = [], defaultdict(list)
    hashes, train_hashes, budgets, unique_runs = set(), set(), set(), set()
    for d in args.evals:
        d = Path(d)
        summary = json.loads((d / "summary.json").read_text())
        provenance = json.loads((d / "provenance.json").read_text())
        hashes.add(provenance["examples_sha256"])
        train_hashes.add(provenance["data_manifest_sha256"])
        c = provenance["config"]
        common_fields = ("d_model", "heads", "d_ff", "max_positions", "train_max_digits",
                         "steps", "micro_batch", "accumulation", "lr", "min_lr", "warmup", "weight_decay",
                         "precision", "grad_clip", "eval_every")
        budgets.add(tuple(c[key] for key in common_fields) + (c["block_layers"] * c["recurrences"],))
        identity = (c["name"], c["seed"])
        if identity in unique_runs:
            raise ValueError("Do not count two checkpoints of one model/seed as independent repeats")
        unique_runs.add(identity)
        if provenance["incomplete_run"]:
            raise ValueError(f"Incomplete training run excluded from comparison: {d}")
        with open(d / "grid.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lengths = sorted({int(r["i"]) for r in rows})
        coords = {n: i for i, n in enumerate(lengths)}
        heat = np.full((len(lengths), len(lengths)), np.nan)
        for r in rows:
            heat[coords[int(r["i"])] , coords[int(r["j"])]] = float(r["accuracy"])
        label = f"{provenance['config']['name']}_seed{provenance['config']['seed']}_step{provenance['step']}"
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(heat, vmin=0, vmax=1, cmap="viridis", origin="upper")
        ax.set_xticks(range(len(lengths)), lengths)
        ax.set_yticks(range(len(lengths)), lengths)
        ax.set(xlabel="Operand B digits (sampled lengths; nonuniform axis)",
               ylabel="Operand A digits", title=label + "\nSparse grid; unsampled cells NOT interpolated")
        bound = sum(n <= summary["train_max_digits"] for n in lengths)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-0.5, -0.5), bound, bound, fill=False, edgecolor="red", linewidth=2))
        fig.colorbar(im, ax=ax, label="Autoregressive exact-match accuracy")
        fig.tight_layout()
        fig.savefig(out / f"{label}.png", dpi=180)
        plt.close(fig)
        row = {"model": provenance["config"]["name"], "seed": provenance["config"]["seed"],
               "step": provenance["step"], "ID_macro": summary["ID"]["macro_cell_accuracy"],
               "OOD_macro": summary["OOD"]["macro_cell_accuracy"],
               "ID_n": summary["ID"]["n"], "OOD_n": summary["OOD"]["n"]}
        all_summary.append(row)
        grouped[row["model"]].append(row)
    if len(hashes) != 1:
        raise ValueError("Comparisons require IDENTICAL test set hashes")
    if len(train_hashes) != 1:
        raise ValueError("Comparisons require IDENTICAL training data manifests")
    if len(budgets) != 1:
        raise ValueError("Comparisons require matched model width, effective depth and training budget")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary[0]))
        w.writeheader()
        w.writerows(all_summary)
    stats = []
    for name, rows in grouped.items():
        values = [r["OOD_macro"] for r in rows if r["OOD_macro"] is not None]
        stats.append({"model": name, "seeds": len(rows),
                      "OOD_macro_mean": float(np.mean(values)) if values else None,
                      "OOD_macro_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                      "warning": "two seeds are preliminary, not strong significance evidence" if len(values) == 2 else "single-seed result"})
    (out / "seed_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
