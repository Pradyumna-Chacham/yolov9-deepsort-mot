from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas import CaptionSegment


class BaseCaptioner(ABC):
    @abstractmethod
    def generate(self, segment: CaptionSegment) -> str:
        raise NotImplementedError