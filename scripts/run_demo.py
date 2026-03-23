from __future__ import annotations

import argparse
from pathlib import Path

from src.captioning.template_captioner import TemplateCaptioner
from src.config import load_config
from src.detectors.factory import get_detector
from src.pipeline.orchestrator import PipelineOrchestrator
from src.trackers.factory import get_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default="demo/sample_videos/sample.mp4")
    parser.add_argument("--output", type=str, default="demo/sample_outputs/output.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    detector = get_detector(cfg)
    tracker = get_tracker(cfg)
    captioner = TemplateCaptioner()

    orchestrator = PipelineOrchestrator(
        config=cfg,
        detector=detector,
        tracker=tracker,
        captioner=captioner,
    )

    Path("demo/sample_outputs").mkdir(parents=True, exist_ok=True)

    result = orchestrator.run(
        input_video_path=args.input,
        output_video_path=args.output,
    )

    print("Run complete.")
    print(f"Output video: {result.output_video_path}")
    print(f"Frames per second: {result.fps:.2f}")
    print(f"Processing FPS: {result.processing_fps:.2f}")
    print(f"Unique track IDs: {result.stats['unique_ids']}")


if __name__ == "__main__":
    main()
