from src.schemas import Detection, Event, PipelineResult


def test_detection_schema() -> None:
    det = Detection(
        frame_index=1,
        bbox=[10.0, 20.0, 30.0, 40.0],
        class_id=0,
        class_name="person",
        confidence=0.9,
    )
    assert det.frame_index == 1
    assert det.class_name == "person"


def test_event_schema() -> None:
    event = Event(
        event_type="enter",
        track_ids=[1],
        start_frame=0,
        end_frame=5,
        timestamp=0.2,
        metadata={"direction": "left"},
    )
    assert event.event_type == "enter"
    assert event.metadata["direction"] == "left"


def test_pipeline_result_schema() -> None:
    result = PipelineResult(
        output_video_path="out.mp4",
        tracks=[],
        events=[],
        segments=[],
        full_summary="No activity detected.",
        fps=25.0,
        processing_fps=7.5,
        stats={"unique_ids": 0},
    )
    assert result.fps == 25.0
    assert result.stats["unique_ids"] == 0