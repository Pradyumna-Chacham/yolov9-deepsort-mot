from __future__ import annotations

from collections import defaultdict

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.schemas import Detection, Track
from src.trackers.base import BaseTracker
from src.utils.bbox_utils import bbox_center, tlwh_to_xyxy, xyxy_to_tlwh


class DeepSortTracker(BaseTracker):
    def __init__(self, max_age: int = 30, n_init: int = 3, embedder: str = "mobilenet") -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.embedder = embedder
        self.current_frame_index = -1
        self.history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.track_first_frame: dict[int, int] = {}
        self.track_metadata: dict[int, dict[str, float | int | str]] = {}
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder=embedder,
        )

    def reset(self) -> None:
        self.current_frame_index = -1
        self.history.clear()
        self.track_first_frame.clear()
        self.track_metadata.clear()
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            embedder=self.embedder,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        if detections:
            self.current_frame_index = max(det.frame_index for det in detections)
        else:
            self.current_frame_index += 1

        ds_detections = []
        detection_metadata = []
        for det in detections:
            tlwh = xyxy_to_tlwh(det.bbox)
            ds_detections.append((tlwh, det.confidence, det.class_name))
            detection_metadata.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "frame_index": det.frame_index,
                }
            )

        tracks = self.tracker.update_tracks(ds_detections, frame=frame, others=detection_metadata)

        output: list[Track] = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue

            ltrb = trk.to_ltrb()
            bbox = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            track_id = int(trk.track_id)
            center = bbox_center(bbox)
            self.history[track_id].append(center)
            det_info = trk.get_det_supplementary()
            if det_info is not None:
                self.track_metadata[track_id] = {
                    "class_id": int(det_info["class_id"]),
                    "class_name": str(det_info["class_name"]),
                    "confidence": float(det_info["confidence"]),
                }

            metadata = self.track_metadata.get(
                track_id,
                {
                    "class_id": -1,
                    "class_name": trk.get_det_class() or "object",
                    "confidence": float(trk.get_det_conf() or 0.0),
                },
            )

            current_frame = self.current_frame_index
            self.track_first_frame.setdefault(track_id, current_frame)

            output.append(
                Track(
                    track_id=track_id,
                    frame_index=current_frame,
                    bbox=bbox,
                    class_id=int(metadata["class_id"]),
                    class_name=str(metadata["class_name"]),
                    confidence=float(metadata["confidence"]),
                    state="moving",
                    trajectory=self.history[track_id].copy(),
                    first_frame=self.track_first_frame[track_id],
                    last_frame=current_frame,
                    age=len(self.history[track_id]),
                )
            )

        return output
