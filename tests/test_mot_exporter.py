from pathlib import Path

from src.io.mot_exporter import MOTExporter
from src.schemas import Track


def test_mot_exporter_writes_expected_format(tmp_path: Path) -> None:
    exporter = MOTExporter()
    tracks = [
        Track(
            track_id=7,
            frame_index=4,
            bbox=[10.0, 20.0, 40.0, 60.0],
            class_id=0,
            class_name="person",
            confidence=0.9123,
            state="moving",
        ),
        Track(
            track_id=3,
            frame_index=1,
            bbox=[1.5, 2.5, 11.5, 12.5],
            class_id=2,
            class_name="car",
            confidence=0.5,
            state="moving",
        ),
    ]

    output_path = tmp_path / "tracked.mp4"
    mot_path = exporter.export(tracks, str(output_path))

    assert mot_path.endswith("tracked.mot.txt")
    content = Path(mot_path).read_text(encoding="utf-8").splitlines()
    assert content == [
        "2,3,1.50,2.50,10.00,10.00,0.5000,-1,-1,-1",
        "5,7,10.00,20.00,30.00,40.00,0.9123,-1,-1,-1",
    ]


def test_mot_exporter_keeps_latest_duplicate_frame_and_id(tmp_path: Path) -> None:
    exporter = MOTExporter()
    tracks = [
        Track(
            track_id=16,
            frame_index=3,
            bbox=[356.30, 109.32, 411.76, 279.02],
            class_id=0,
            class_name="person",
            confidence=0.2864,
            state="moving",
        ),
        Track(
            track_id=16,
            frame_index=3,
            bbox=[355.72, 108.15, 412.38, 281.54],
            class_id=0,
            class_name="person",
            confidence=0.2864,
            state="moving",
        ),
    ]

    mot_path = exporter.export(tracks, str(tmp_path / "tracked.mp4"))
    content = Path(mot_path).read_text(encoding="utf-8").splitlines()

    assert content == [
        "4,16,355.72,108.15,56.66,173.39,0.2864,-1,-1,-1",
    ]
