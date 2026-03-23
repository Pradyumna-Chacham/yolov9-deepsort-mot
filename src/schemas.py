from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    frame_index: int
    bbox: list[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


@dataclass
class Track:
    track_id: int
    frame_index: int
    bbox: list[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    state: str  # "moving" | "stopped" | "occluded"
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    age: int = 0


@dataclass
class Event:
    event_type: str  # enter|exit|stop|move|occlusion|crossing
    track_ids: list[int]
    start_frame: int
    end_frame: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptionSegment:
    start_time: float
    end_time: float
    events: list[Event]
    template_caption: str
    vlm_caption: str | None = None
    fused_summary: str | None = None


@dataclass
class PipelineResult:
    output_video_path: str
    tracks: list[Track]
    events: list[Event]
    segments: list[CaptionSegment]
    full_summary: str
    fps: float
    processing_fps: float
    stats: dict[str, Any] = field(default_factory=dict)