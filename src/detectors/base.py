from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import Detection


class BaseDetector(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError