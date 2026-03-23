from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int) -> None:
        self.output_path = str(output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
