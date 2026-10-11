import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from abacus.config import preset
from abacus.data import (BOS, EOS, PAD, ENC, BatchStream, collate, encode, example,
                         finite_pool, prepare, read_jsonl, split_of)
from abacus.evaluation import generate, wilson
from abacus.model import ArithmeticTransformer, FIRE, answer_loss, local_positions

torch.set_num_threads(1)


def tiny(name="abacus"):
    return replace(preset(name), d_model=32, d_ff=64, heads=4, block_layers=2,
                   train_max_digits=2, steps=10, micro_batch=4, accumulation=1)


def test_span_positions_and_non_digits():
    x = torch.tensor([[BOS] + encode("321+54=861") + [EOS, PAD]])
    assert local_positions(x).tolist() == [[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 0]]


def test_official_abacus_equivalence():
    from pathlib import Path
    ns = {"torch": torch, "random": random}
    source = Path("reference/abacus_official.py")
    if not source.exists():
        pytest.skip("Official source snapshot not included")
    exec(compile(source.read_text(), str(source), "exec"), ns)
    official = ns["Abacus"](list(range(ENC["0"], ENC["9"] + 1)), 32)
    ids = torch.tensor([[BOS] + encode("321+54=861") + [EOS]])
    mask = (ids >= ENC["0"]) & (ids <= ENC["9"])
    assert torch.equal(local_positions(ids), official.helper(mask, ids.device))
    ours = ArithmeticTransformer(tiny()).eval()
    ours.abacus.weight.data.copy_(official.embedding.weight[:512])
    official.eval()
    assert torch.equal(ours.abacus(local_positions(ids)), official(ids))


def test_answer_shift_mask():
    seq = [BOS] + encode("321+54=861") + [EOS]
    row = np.array(seq + [PAD] * 3, dtype=np.uint8)
    x, y = collate([row], "cpu")
    eq = seq.index(ENC["="])
    assert y[0, :eq].eq(-100).all()
    assert y[0, eq:].tolist() == encode("861") + [EOS]
    assert x[0, eq].item() == ENC["="]


def test_commuted_examples_are_same_split():
    for a, b in [(3, 5), (12, 41), (123456, 9)]:
        assert split_of(a, b) == split_of(b, a)
    pools = [set(finite_pool(1, 1, s)) for s in ("train", "val", "test")]
    assert all(pools)
    assert not (pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])
    assert len(set.union(*pools)) == 81


def test_dataset_generation_checksums_and_order_resume(tmp_path):
    root = tmp_path / "data"
    manifest = prepare(root, samples=400, train_max=2, per_cell=5, test_lengths=[1, 2, 3])
    assert manifest["width"] == 11
    data = np.load(root / "train.npy")
    assert data.shape == (400, 11)
    for s in ("val", "test"):
        assert all(split_of(int(x["a"]), int(x["b"])) == s for x in read_jsonl(root / f"{s}.jsonl"))
    stream = BatchStream(root, 7, 42)
    stream.next()
    recovered = BatchStream(root, 7, 42, stream.state())
    assert np.array_equal(stream.next(), recovered.next())
    with pytest.raises(FileExistsError):
        prepare(root, samples=400, train_max=2)


@pytest.mark.parametrize("name", ["nope", "ape", "fire", "abacus", "abacus_ii", "abacus_loop", "abacus_loop_prog", "abacus_fire"])
def test_cache_equals_full_context_and_causality(name):
    m = ArithmeticTransformer(tiny(name)).eval()
    x = torch.tensor([[BOS] + encode("21+7=91")])
    with torch.no_grad():
        full, _ = m(x)
        prefix, cache = m(x[:, :-2], use_cache=True)
        pos = local_positions(x)
        for k in range(x.size(1) - 2, x.size(1)):
            inc, cache = m(x[:, k:k + 1], positions=pos[:, k:k + 1], cache=cache, use_cache=True)
            assert torch.allclose(inc[:, -1], full[:, k], atol=2e-6, rtol=1e-5)
        changed = x.clone()
        changed[:, -1] = ENC["0"]
        after, _ = m(changed)
        assert torch.allclose(full[:, :-1], after[:, :-1], atol=2e-6, rtol=1e-5)
    assert len(cache) == m.cfg.block_layers * m.cfg.recurrences


def test_eval_offset_guard_and_common_init():
    a = ArithmeticTransformer(tiny("abacus"))
    b = ArithmeticTransformer(tiny("nope"))
    assert torch.equal(a.blocks[0].attn.qkv.weight, b.blocks[0].attn.qkv.weight)
    assert torch.equal(a.tokens.weight, b.tokens.weight)
    a.eval()
    with pytest.raises(ValueError):
        a(torch.tensor([[BOS, ENC["1"]]]), beta=42)


def test_fire_cached_global_query_position():
    f = FIRE(4)
    full = f(12, 12, 0, "cpu")
    last = f(1, 12, 11, "cpu")
    assert torch.allclose(last, full[:, :, -1:, :])


def test_fire_continuous_formula_matches_official_path():
    f = FIRE(4)
    p = torch.arange(10, dtype=torch.float)
    rel = torch.log(torch.abs(f.c * (p[:, None] - p[None, :])) + 1)
    norm = torch.log(torch.abs(f.c * torch.maximum(p, torch.abs(f.L_multiplier * f.init_L))) + 1)[:, None]
    official_formula = f.mlp((rel / (norm + 1e-6)).unsqueeze(-1)).unsqueeze(0).permute(0, 3, 1, 2)
    assert torch.allclose(f(10, 10, 0, "cpu"), official_formula, atol=1e-7, rtol=1e-5)


def test_gradient_flows_through_full_recurrences():
    m = ArithmeticTransformer(tiny("abacus_loop"))
    row = [BOS] + encode("21+7=91") + [EOS]
    x, y = collate([np.array(row, dtype=np.uint8)], "cpu")
    logits, _ = m(x, beta=65)
    loss = answer_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.abacus.weight.grad is not None
    assert m.blocks[-1].attn.qkv.weight.grad.abs().sum() > 0


def test_autoregressive_cache_matches_uncached():
    m = ArithmeticTransformer(tiny())
    a = generate(m, ["21+7=", "34+8="], 4, "cpu", use_cache=True)
    b = generate(m, ["21+7=", "34+8="], 4, "cpu", use_cache=False)
    assert a == b


def test_tiny_overfit_then_greedy_decode():
    # This verifies executable learning mechanics, NOT ID/OOD research accuracy.
    m = ArithmeticTransformer(replace(tiny(), block_layers=1))
    opt = torch.optim.AdamW(m.parameters(), lr=0.003)
    row = np.array([BOS] + encode("21+7=91") + [EOS], dtype=np.uint8)
    x, y = collate([row] * 4, "cpu")
    for _ in range(160):
        m.train()
        opt.zero_grad()
        logits, _ = m(x, beta=1)
        loss = answer_loss(logits, y)
        loss.backward()
        opt.step()
    pred = generate(m, ["21+7="], 4, "cpu")[0]
    assert pred == {"prediction": "91", "ended": True}


def test_wilson_not_fake_exact_certainty():
    lo, hi = wilson(64, 64)
    assert lo < 1 and hi <= 1


def test_default_20_digit_model_and_max_beta_carry_backward():
    m = ArithmeticTransformer(preset("abacus"))
    # (10^20-1)+1 gives a 21-digit result and the highest trained index120.
    s = "9" * 20 + "+1=" + "0" * 20 + "1"
    row = np.array([BOS] + encode(s) + [EOS], dtype=np.uint8)
    x, y = collate([row], "cpu")
    logits, _ = m(x, beta=100)
    assert logits.shape == (*x.shape, 15)
    loss = answer_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.abacus.weight.grad[120].abs().sum() > 0


def test_full_100_digit_decoding_context_boundary():
    m = ArithmeticTransformer(tiny()).eval()
    prefix_and_answer = "9" * 100 + "+" + "8" * 100 + "=" + "1" * 101
    ids = torch.tensor([[BOS] + encode(prefix_and_answer)])
    assert ids.size(1) == 304
    with torch.no_grad():
        logits, _ = m(ids)
    assert logits.shape == (1, 304, 15)


def test_weighted_accumulation_matches_global_token_mean():
    m = ArithmeticTransformer(tiny())
    rows = [np.array([BOS] + encode(s) + [EOS] + [PAD] * (12 - len(s)), dtype=np.uint8)
            for s in ("1+2=3", "99+99=891")]
    x, y = collate(rows, "cpu")
    logits, _ = m(x)
    answer_loss(logits, y).backward()
    direct = m.tokens.weight.grad.clone()
    m.zero_grad()
    counts = [len(s.split("=")[1]) + 1 for s in ("1+2=3", "99+99=891")]
    for row, count in zip(rows, counts):
        xx, yy = collate([row], "cpu")
        ll, _ = m(xx)
        (answer_loss(ll, yy) * count / sum(counts)).backward()
    assert torch.allclose(direct, m.tokens.weight.grad, atol=1e-6, rtol=1e-5)


def test_full_training_resume_exactly_matches_uninterrupted(tmp_path):
    from abacus.training import train
    root = tmp_path / "data"
    prepare(root, samples=40, train_max=2, per_cell=2, test_lengths=[1, 2])
    cfg = replace(tiny(), steps=6, warmup=2, eval_every=3, save_every=2, accumulation=2)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert train(cfg, root, full, torch.device("cpu"))["completed"]
    status = train(cfg, root, resumed, torch.device("cpu"), max_hours=0)
    assert not status["completed"] and status["step"] == 1
    assert train(cfg, root, resumed, torch.device("cpu"), resume=resumed / "last.pt")["completed"]
    aa = torch.load(full / "last.pt", weights_only=False)
    bb = torch.load(resumed / "last.pt", weights_only=False)
    assert aa["stream"] == bb["stream"]
    assert aa["method_rng"] == bb["method_rng"]
    assert torch.equal(aa["torch_rng"], bb["torch_rng"])
    for name in aa["model"]:
        assert torch.equal(aa["model"][name], bb["model"][name]), name
    for index in aa["optimizer"]["state"]:
        for key in aa["optimizer"]["state"][index]:
            assert torch.equal(aa["optimizer"]["state"][index][key], bb["optimizer"]["state"][index][key])
    with pytest.raises(ValueError, match="Dataset changed"):
        with open(root / "val.jsonl", "a") as f:
            f.write("\n")
        train(cfg, root, resumed, torch.device("cpu"), resume=resumed / "last.pt")
