"""Real CUDA tests; CPU-only CI skips them instead of claiming GPU validation."""
from dataclasses import replace

import pytest
import torch

from abacus.config import preset
from abacus.data import BOS, EOS, encode
from abacus.model import ArithmeticTransformer, answer_loss, local_positions
from abacus.evaluation import amp_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("name", ["nope", "fire", "abacus_fixed", "abacus", "abacus_ii", "abacus_loop"])
def test_cuda_default_model_backward_and_cached_logits(name):
    cfg = replace(preset(name), micro_batch=2)
    model = ArithmeticTransformer(cfg).cuda().train()
    ids = torch.tensor([[BOS] + encode("9" * 20 + "+" + "9" * 20 + "=" + "8" + "9" * 19 + "1") + [EOS]], device="cuda")
    x, y = ids[:, :-1], ids[:, 1:].clone()
    y[:, :42] = -100
    with amp_context(torch.device("cuda")):
        logits, _ = model(x, beta=cfg.offset_k)
        loss = answer_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    # FP32 cache equivalence isolates indexing from mixed-precision roundoff.
    with torch.no_grad():
        full, _ = model(x)
        _, cache = model(x[:, :-1], use_cache=True)
        last, _ = model(x[:, -1:], positions=local_positions(x)[:, -1:], cache=cache, use_cache=True)
    assert torch.allclose(full[:, -1:], last, atol=2e-4, rtol=2e-4)


def test_cuda_100_digit_context_and_cache():
    model = ArithmeticTransformer(preset("abacus")).cuda().eval()
    ids = torch.tensor([[BOS] + encode("9" * 100 + "+" + "8" * 100 + "=" + "1" * 101)], device="cuda")
    assert ids.size(1) == 304
    with torch.no_grad(), amp_context(torch.device("cuda")):
        full, _ = model(ids)
        _, cache = model(ids[:, :-1], use_cache=True)
        last, _ = model(ids[:, -1:], positions=local_positions(ids)[:, -1:], cache=cache, use_cache=True)
    assert torch.isfinite(full).all() and torch.isfinite(last).all()
    assert torch.allclose(full[:, -1:].float(), last.float(), atol=0.03, rtol=0.03)
