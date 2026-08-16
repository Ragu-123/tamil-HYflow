import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Config:
    sample_rate: int = 24000
    latent_rate: int = 25
    latent_dim: int = 64
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-2
    max_seconds: float = 12.0
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = True
    save_dir: str = "checkpoints"

    @classmethod
    def from_json(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_json(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
