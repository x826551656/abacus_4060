"""Small pre-LN decoder. Abacus/FIRE semantics follow the paper; see DEVIATIONS.md."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .data import DIGIT_MIN, DIGIT_MAX, VOCAB_SIZE


def local_positions(ids):
    """1-based digit-span positions; every non-digit has index zero."""
    mask = (ids >= DIGIT_MIN) & (ids <= DIGIT_MAX)
    index = torch.arange(ids.size(1), device=ids.device).expand_as(ids)
    prev = F.pad(mask[:, :-1], (1, 0), value=False)
    starts = mask & ~prev
    start_index = torch.where(starts, index, torch.zeros_like(index))
    last_start = start_index.cummax(dim=1).values
    return torch.where(mask, index - last_start + 1, torch.zeros_like(index))


class FIRE(nn.Module):
    """Unrandomized FIRE: signed causal distance becomes abs distance in log map.

    Matches the official FIRE with max_length=0 (no random-position subsampling).
    The denominator belongs to the query's GLOBAL position, not key position.
    """
    def __init__(self, heads, width=32):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.1))
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("init_L", torch.tensor(512.0))
        self.mlp = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, heads))

    def forward(self, qlen, klen, past, device):
        qpos = torch.arange(past, past + qlen, device=device, dtype=torch.float32)
        kpos = torch.arange(klen, device=device, dtype=torch.float32)
        distances = (qpos[:, None] - kpos[None, :]).abs()
        threshold = (self.L_multiplier * self.init_L).abs()
        denominator = torch.log1p((self.c * torch.maximum(qpos, threshold)).abs())[:, None]
        normalized = torch.log1p((self.c * distances).abs()) / (denominator + 1e-6)
        return self.mlp(normalized.unsqueeze(-1)).permute(2, 0, 1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.d_model // cfg.heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.fire = FIRE(cfg.heads) if "fire" in cfg.positional else None

    def forward(self, x, cache=None, use_cache=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [z.view(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        past = 0
        if cache is not None:
            past = cache[0].size(2)
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        if self.fire is None and past == 0:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        elif self.fire is None and t == 1:
            # The only query is the newest token: all cached keys are legal.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            qpos = torch.arange(past, past + t, device=x.device)
            kpos = torch.arange(k.size(2), device=x.device)
            legal = kpos[None, :] <= qpos[:, None]
            bias = torch.zeros((1, 1, t, k.size(2)), device=x.device, dtype=q.dtype)
            if self.fire is not None:
                bias = self.fire(t, k.size(2), past, x.device).to(q.dtype)
            bias = bias.masked_fill(~legal[None, None, :, :], float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        out = self.proj(out.transpose(1, 2).contiguous().view(b, t, d))
        return out, (k, v) if use_cache else None


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, cache=None, use_cache=False):
        a, cache = self.attn(self.norm1(x), cache, use_cache)
        x = x + a
        return x + self.ff(self.norm2(x)), cache


class ArithmeticTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.validate()
        self.tokens = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        # Index zero IS learned, as in authors' standalone implementation.
        self.abacus = nn.Embedding(cfg.max_positions, cfg.d_model) if "abacus" in cfg.positional else None
        self.ape = nn.Embedding(cfg.max_positions, cfg.d_model) if cfg.positional == "ape" else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.block_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tokens.weight
        self.reset_parameters(cfg.seed)

    def reset_parameters(self, seed):
        # Common named parameters start identically across NoPE/Abacus variants.
        # Adding an embedding table must not shift all subsequent RNG draws.
        for name, p in self.named_parameters():
            digest = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")
            g = torch.Generator(device="cpu").manual_seed(seed + digest)
            if name.endswith(".c") or name.endswith("L_multiplier"):
                continue
            if p.ndim == 1:
                nn.init.ones_(p) if name.endswith("weight") else nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0, std=0.02, generator=g)
        # Residual-output initialization scaled by EFFECTIVE depth.
        scale = math.sqrt(2 * self.cfg.block_layers * self.cfg.recurrences)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.proj.weight.div_(scale)
                block.ff[2].weight.div_(scale)

    def forward(self, ids, *, beta=1, recurrences=None, positions=None,
                cache=None, use_cache=False):
        r = self.cfg.recurrences if recurrences is None else recurrences
        if not 1 <= r <= self.cfg.recurrences:
            raise ValueError("Invalid recurrence count")
        if not self.training and beta != 1:
            raise ValueError("Evaluation requires beta=1")
        past = 0 if cache is None else cache[0][0].size(2)
        if past + ids.size(1) > self.cfg.max_positions:
            raise ValueError("Sequence exceeds max_positions; do not silently truncate")
        inp = self.tokens(ids)
        if self.abacus is not None:
            if positions is None:
                if past:
                    raise ValueError("Cached decoding requires explicit local positions")
                positions = local_positions(ids)
            index = torch.where(positions > 0, positions + beta - 1, positions)
            if int(index.max()) >= self.cfg.max_positions:
                raise ValueError("Abacus embedding index out of range")
            inp = inp + self.abacus(index)
        if self.ape is not None:
            glob = torch.arange(past, past + ids.size(1), device=ids.device)
            inp = inp + self.ape(glob)[None, :, :]
        x, updated = inp, []
        expected_cache = len(self.blocks) * r
        if cache is not None and len(cache) != expected_cache:
            raise ValueError("Cache must be indexed by effective layer, not shared block")
        for repeat in range(r):
            for k, block in enumerate(self.blocks):
                effective_layer = repeat * len(self.blocks) + k
                if self.cfg.input_injection:
                    x = x + inp
                old = None if cache is None else cache[effective_layer]
                x, current = block(x, old, use_cache)
                if use_cache:
                    updated.append(current)
        return self.head(self.norm(x)), updated if use_cache else None


def answer_loss(logits, labels):
    """Token-mean answer loss; includes EOS, ignores questions and padding."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
