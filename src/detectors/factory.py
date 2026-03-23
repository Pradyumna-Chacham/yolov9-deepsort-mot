from __future__ import annotations

from src.config import AppConfig
from src.detectors.base import BaseDetector
from src.detectors.ultralytics_adapter import UltralyticsAdapter
from src.detectors.yolov9 import YOLOv9Adapter


def get_detector(cfg: AppConfig) -> BaseDetector:
    name = cfg.detector["name"].lower()

    common_kwargs = {
        "weights": cfg.detector["weights"],
        "conf_threshold": float(cfg.detector["conf_threshold"]),
        "iou_threshold": float(cfg.detector["iou_threshold"]),
        "classes": list(cfg.detector["classes"]),
        "device": cfg.device,
    }

    if name == "yolov9":
        return YOLOv9Adapter(**common_kwargs)

    if name == "ultralytics":
        return UltralyticsAdapter(**common_kwargs)

    raise ValueError(f"Unsupported detector: {name}")
