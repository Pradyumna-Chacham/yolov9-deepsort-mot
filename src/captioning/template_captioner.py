from __future__ import annotations

from src.captioning.base import BaseCaptioner
from src.schemas import CaptionSegment


class TemplateCaptioner(BaseCaptioner):
    def generate(self, segment: CaptionSegment) -> str:
        if not segment.events:
            return "No significant events detected in this segment."
        return f"{len(segment.events)} events detected in this segment."
