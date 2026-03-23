from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.base import BaseDetector
from src.schemas import Detection


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLOv9Adapter(BaseDetector):
    def __init__(
        self,
        weights: str,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int],
        device: str,
    ) -> None:
        self.weights = Path(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        self.model = None

    @property
    def name(self) -> str:
        return "yolov9"

    def load(self) -> None:
        """
        Stub for now. In the next step we will replace this with the real YOLOv9 loader.
        """
        self.model = "stub-loaded"

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        # Stub result so the pipeline can be built incrementally.
        _ = frame
        _ = frame_index
        return []