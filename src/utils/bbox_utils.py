from __future__ import annotations


def xyxy_to_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_tlwh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def tlwh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clip_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]