import json
import sys
from pathlib import Path

import pytest
import torch

import budget
import run
import suite
from abacus.config import preset


def test_suite_freezes_full_plan_even_when_executing_subset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "train", lambda c, d, r, dev, resume: calls.append((c.name, c.seed)))
    root = tmp_path / "runs"
    args = ["suite.py", "--output", str(root), "--steps", "12000", "--device", "cpu", "--execute"]
    monkeypatch.setattr(sys, "argv", args + ["--only", "nope_seed42"])
    suite.main()
    plan = json.loads((root / "plan.json").read_text())
    assert len(plan["jobs"]) == 8
    assert calls == [("nope", 42)]
    monkeypatch.setattr(sys, "argv", args + ["--only", "fire_seed42"])
    suite.main()
    assert calls[-1] == ("fire", 42)
    monkeypatch.setattr(sys, "argv", ["suite.py", "--output", str(root), "--steps", "24000",
                                      "--device", "cpu", "--execute", "--only", "abacus_seed42"])
    with pytest.raises(ValueError, match="Frozen plan"):
        suite.main()


def test_early_best_checkpoint_of_completed_run_not_marked_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    ck = root / "best.pt"
    ck.write_bytes(b"unit-test-fixture-not-a-real-checkpoint")
    (root / "status.json").write_text(json.dumps({"completed": True, "step": 24000}))
    ex = tmp_path / "examples.jsonl"
    ex.write_text("")
    class Model:
        cfg = preset("abacus")
    monkeypatch.setattr(run, "load_model", lambda p, dev: (Model(), {"step": 2000, "manifest_hash": "unit-test-data-hash"}))
    def dummy_evaluate(m, rows, dev, batch, precision, output):
        Path(output).mkdir()
        return {}, []
    monkeypatch.setattr(run, "evaluate", dummy_evaluate)
    out = tmp_path / "eval"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "eval", "--checkpoint", "run/best.pt", "--examples", "examples.jsonl",
                                      "--output", "eval", "--device", "cpu"])
    run.main()
    p = json.loads((out / "provenance.json").read_text())
    assert p["step"] == 2000 and p["completed_steps"] == 24000 and not p["incomplete_run"]
    assert p["checkpoint"] == "run/best.pt"


def test_budget_rejects_cpu_measurements(tmp_path, monkeypatch):
    paths = []
    for name in ("nope", "fire", "abacus_fixed", "abacus", "abacus_ii", "abacus_loop"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"config": preset(name).dict(), "environment": {"gpu": None},
                                   "median_seconds_per_update": 0.1}))
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["budget.py", "--benchmarks"] + paths)
    with pytest.raises(ValueError, match="CPU measurements"):
        budget.main()
