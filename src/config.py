from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any]

    @property
    def system(self) -> dict[str, Any]:
        return self.raw.get("system", {})

    @property
    def device(self) -> str:
        requested = self.raw.get("device", "auto")
        if requested != "auto":
            return requested

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def detector(self) -> dict[str, Any]:
        return self.raw["detector"]

    @property
    def tracker(self) -> dict[str, Any]:
        return self.raw["tracker"]

    @property
    def events(self) -> dict[str, Any]:
        return self.raw["events"]

    @property
    def captioning(self) -> dict[str, Any]:
        return self.raw["captioning"]

    @property
    def visualization(self) -> dict[str, Any]:
        return self.raw["visualization"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]

    @property
    def demo(self) -> dict[str, Any]:
        return self.raw["demo"]

    @property
    def seed(self) -> int:
        return int(self.system.get("seed", 42))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig(raw=raw)
    set_seed(cfg.seed)
    return cfg