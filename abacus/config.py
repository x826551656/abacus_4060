from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Config:
    name: str = "abacus"
    d_model: int = 256
    heads: int = 8
    d_ff: int = 512
    block_layers: int = 8
    recurrences: int = 1
    positional: str = "abacus"  # nope, ape, fire, abacus, abacus_fire
    offset_k: int = 100  # beta in [1,k], not shift in [0,k]
    max_positions: int = 512
    input_injection: bool = False
    progressive_alpha: float = 0.0  # weight of random-short-recurrence loss
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup: int = 1000
    steps: int = 24000
    micro_batch: int = 32
    accumulation: int = 4
    grad_clip: float = 1.0
    seed: int = 42
    train_max_digits: int = 20
    precision: str = "auto"
    eval_every: int = 2000
    save_every: int = 1000

    def validate(self):
        assert self.d_model % self.heads == 0
        assert self.block_layers >= 1 and self.recurrences >= 1
        assert self.steps >= 1 and self.micro_batch >= 1 and self.accumulation >= 1
        assert self.eval_every >= 1 and self.save_every >= 1
        assert 0 < self.min_lr <= self.lr and self.warmup >= 0
        assert self.max_positions >= 3 * self.train_max_digits + 4
        assert 0 <= self.progressive_alpha <= 1
        assert self.progressive_alpha == 0 or self.recurrences > 1
        assert self.offset_k >= 1
        assert self.max_positions > self.offset_k + self.train_max_digits
        assert self.positional in {"nope", "ape", "fire", "abacus", "abacus_fire"}
        return self

    def dict(self):
        return asdict(self)


def preset(name: str, seed=42):
    c = Config(name=name, seed=seed)
    if name in {"nope", "ape", "fire"}:
        c = replace(c, positional=name)
    elif name == "abacus_fixed":
        c = replace(c, offset_k=1)
    elif name == "abacus_ii":
        c = replace(c, input_injection=True)
    elif name == "abacus_loop":
        c = replace(c, input_injection=True, block_layers=4, recurrences=2)
    elif name == "abacus_loop_prog":
        c = replace(c, input_injection=True, block_layers=4, recurrences=2,
                    progressive_alpha=0.5)
    elif name == "abacus_fire":
        c = replace(c, positional="abacus_fire")
    elif name != "abacus":
        raise ValueError(f"Unknown preset {name}")
    return c.validate()


PRESETS = ("nope", "fire", "abacus_fixed", "abacus", "abacus_ii",
           "abacus_loop", "abacus_loop_prog", "ape", "abacus_fire")
