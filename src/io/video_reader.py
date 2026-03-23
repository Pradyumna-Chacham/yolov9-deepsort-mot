from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path: str) -> None:
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1

    def release(self) -> None:
        self.cap.release()
