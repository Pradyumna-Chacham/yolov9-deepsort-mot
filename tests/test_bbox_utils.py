from src.utils.bbox_utils import (
    bbox_center,
    tlwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_tlwh,
    xyxy_to_xywh,
)


def test_xyxy_to_xywh() -> None:
    assert xyxy_to_xywh([10, 20, 30, 50]) == [10, 20, 20, 30]


def test_xywh_to_xyxy() -> None:
    assert xywh_to_xyxy([10, 20, 20, 30]) == [10, 20, 30, 50]


def test_xyxy_to_tlwh() -> None:
    assert xyxy_to_tlwh([5, 6, 15, 26]) == [5, 6, 10, 20]


def test_tlwh_to_xyxy() -> None:
    assert tlwh_to_xyxy([5, 6, 10, 20]) == [5, 6, 15, 26]


def test_bbox_center() -> None:
    assert bbox_center([0, 0, 10, 20]) == (5.0, 10.0)