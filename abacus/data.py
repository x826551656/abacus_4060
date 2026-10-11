"""Exact digit lengths, answer-only labels, and order-invariant split isolation."""
import hashlib
import itertools
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

PAD, BOS, EOS = 0, 1, 2
CHARS = "0123456789+="
ENC = {c: i + 3 for i, c in enumerate(CHARS)}
DEC = {i: c for c, i in ENC.items()}
VOCAB_SIZE = 3 + len(CHARS)
DIGIT_MIN, DIGIT_MAX = ENC["0"], ENC["9"]
SPLIT_SALT = "abacus4060-v1-split"


def encode(s):
    return [ENC[c] for c in s]


def decode(ids):
    return "".join(DEC.get(int(i), f"<{int(i)}>") for i in ids)


def interval(length):
    # Positive integers, consistent with the default exact-length official generator.
    return 10 ** (length - 1), 10 ** length - 1


def split_of(a, b):
    # Treat a+b and b+a as the SAME item. Fix salt across all models and seeds.
    lo, hi = sorted((a, b))
    h = hashlib.blake2b(f"{SPLIT_SALT}:{lo}:{hi}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(h, "little") % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def example(a, b):
    prompt = f"{str(a)[::-1]}+{str(b)[::-1]}="
    answer = str(a + b)[::-1]
    return {"a": str(a), "b": str(b), "i": len(str(a)), "j": len(str(b)),
            "prompt": prompt, "answer": answer}


@lru_cache(maxsize=128)
def finite_pool(i, j, split):
    alo, ahi = interval(i)
    blo, bhi = interval(j)
    return [(a, b) for a in range(alo, ahi + 1) for b in range(blo, bhi + 1)
            if split_of(a, b) == split]


def small_cell(i, j):
    return 81 * 10 ** (i + j - 2) <= 20000


def draw_pair(rng, i, j, split):
    if small_cell(i, j):
        pool = finite_pool(i, j, split)
        if not pool:
            raise ValueError(f"Empty split cell {(i, j, split)}")
        return rng.choice(pool)
    for _ in range(10000):
        a, b = rng.randint(*interval(i)), rng.randint(*interval(j))
        if split_of(a, b) == split:
            return a, b
    raise RuntimeError("Could not sample split")


def eval_examples(lengths, per_cell, split, seed):
    rng = random.Random(seed)
    out = []
    for i, j in itertools.product(lengths, repeat=2):
        if small_cell(i, j):
            pool = finite_pool(i, j, split).copy()
            rng.shuffle(pool)
            pairs = pool[:per_cell]  # Do not fake n=64 by repeating nine tiny examples.
        else:
            pairs, seen = [], set()
            while len(pairs) < per_cell:
                a, b = draw_pair(rng, i, j, split)
                if (a, b) not in seen:
                    pairs.append((a, b))
                    seen.add((a, b))
        out.extend(example(a, b) for a, b in pairs)
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root, samples=200_000, train_max=20, seed=2026, per_cell=64,
            test_lengths=None):
    if samples < 1 or train_max < 1 or per_cell < 1:
        raise ValueError("samples, train_max and per_cell must be positive")
    if test_lengths is not None and (not test_lengths or any(n < 1 for n in test_lengths)):
        raise ValueError("Test digit lengths must be positive")
    if test_lengths is not None:
        test_lengths = sorted(set(test_lengths))
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {root}")
    root.mkdir(parents=True, exist_ok=True)
    cells = train_max ** 2
    if samples % cells:
        raise ValueError(f"samples must be divisible by {cells}, for exact stratification")
    # BOS + A + '+' + B + '=' + SUM + EOS => 3*L+5 tokens maximum.
    width = 3 * train_max + 5
    ids = np.lib.format.open_memmap(root / "train.npy", mode="w+", dtype=np.uint8,
                                    shape=(samples, width))
    rng = random.Random(seed)
    for idx in range(samples):
        cell = idx % cells
        i, j = cell // train_max + 1, cell % train_max + 1
        a, b = draw_pair(rng, i, j, "train")
        ex = example(a, b)
        row = [BOS] + encode(ex["prompt"] + ex["answer"]) + [EOS]
        ids[idx] = PAD
        ids[idx, :len(row)] = row
        if idx and idx % 250000 == 0:
            print(f"generated {idx:,}/{samples:,}", flush=True)
    ids.flush()
    del ids
    val_lengths = sorted(set([1, 2, 4, 8, 12, 16, train_max]) & set(range(1, train_max + 1)))
    write_jsonl(root / "val.jsonl", eval_examples(val_lengths, 8, "val", seed + 1))
    test_lengths = test_lengths or sorted(set(val_lengths + [24, 32, 40, 60, 80, 100]))
    write_jsonl(root / "test.jsonl", eval_examples(test_lengths, per_cell, "test", seed + 2))
    manifest = {"schema": 1, "samples": samples, "train_max": train_max,
                "seed": seed, "width": width, "split_salt": SPLIT_SALT,
                "split_ratio": "80/10/10 by canonical operand pair hash",
                "test_lengths": test_lengths, "test_requested_per_cell": per_cell,
                "train_sampling": "with replacement, exactly equal length-pair counts",
                "files": {f: sha256(root / f) for f in ("train.npy", "val.jsonl", "test.jsonl")}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def collate(rows, device):
    x = torch.as_tensor(np.stack(rows).astype(np.int64), device=device)
    last = int((x != PAD).sum(1).max().item())
    x = x[:, :last]
    labels = x[:, 1:].clone()
    equals = (x == ENC["="]).long().argmax(1)
    positions = torch.arange(labels.size(1), device=device)[None, :]
    # Logits at '=' predict the FIRST answer digit. Do not shift this mask twice.
    labels[(positions < equals[:, None]) | (labels == PAD)] = -100
    return x[:, :-1], labels


class BatchStream:
    """Deterministic sample order; resume needs only epoch+cursor, not a giant RNG dump."""
    def __init__(self, root, batch, seed, state=None):
        self.data = np.load(Path(root) / "train.npy", mmap_mode="r")
        self.batch, self.seed = batch, seed
        self.epoch = 0 if state is None else state["epoch"]
        self.cursor = 0 if state is None else state["cursor"]
        self._order()

    def _order(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.data))

    def next(self):
        chunks, remaining = [], self.batch
        while remaining:
            n = min(remaining, len(self.data) - self.cursor)
            ix = self.order[self.cursor:self.cursor + n]
            chunks.append(np.asarray(self.data[ix]))
            self.cursor += n
            remaining -= n
            if self.cursor == len(self.data):
                self.epoch += 1
                self.cursor = 0
                self._order()
        return np.concatenate(chunks)

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}
