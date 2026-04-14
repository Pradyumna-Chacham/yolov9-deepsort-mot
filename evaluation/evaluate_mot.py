from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import motmetrics as mm
import pandas as pd


PRED_COLUMNS = [
    "FrameId",
    "Id",
    "X",
    "Y",
    "Width",
    "Height",
    "Confidence",
    "ClassId",
    "Visibility",
    "Unused",
]
GT_COLUMNS = [
    "FrameId",
    "Id",
    "X",
    "Y",
    "Width",
    "Height",
    "Confidence",
    "ClassId",
    "Visibility",
]
SUMMARY_METRICS = [
    "num_frames",
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "num_switches",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_objects",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MOT-format prediction file against a MOTChallenge sequence. "
            "By default, only the overlapping frame range is scored."
        )
    )
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted .mot.txt file")
    parser.add_argument(
        "--sequence-dir",
        type=str,
        required=True,
        help="Path to MOT sequence directory containing gt/gt.txt and seqinfo.ini",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=None,
        help="Optional hard cap on evaluated duration in seconds",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="1-based first frame to evaluate",
    )
    return parser.parse_args()


def load_seqinfo(path: Path) -> tuple[float, int]:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    section = parser["Sequence"]
    fps = float(section.get("frameRate", "0"))
    seq_length = int(section.get("seqLength", "0"))
    return fps, seq_length


def load_prediction(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=PRED_COLUMNS)


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, header=None, names=GT_COLUMNS)
    # Standard MOT pedestrian evaluation filter.
    return gt[(gt["Confidence"] == 1) & (gt["ClassId"] == 1)].copy()


def determine_end_frame(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    seq_length: int,
    fps: float,
    sample_seconds: float | None,
) -> int:
    max_frame = min(
        int(pred["FrameId"].max()) if not pred.empty else 0,
        int(gt["FrameId"].max()) if not gt.empty else 0,
        seq_length,
    )
    if sample_seconds is not None:
        sample_end = int(sample_seconds * fps)
        max_frame = min(max_frame, sample_end)
    return max_frame


def accumulate_metrics(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in range(start_frame, end_frame + 1):
        gt_frame = gt[gt["FrameId"] == frame_id]
        pred_frame = pred[pred["FrameId"] == frame_id]

        gt_ids = gt_frame["Id"].tolist()
        pred_ids = pred_frame["Id"].tolist()
        gt_boxes = gt_frame[["X", "Y", "Width", "Height"]].to_numpy()
        pred_boxes = pred_frame[["X", "Y", "Width", "Height"]].to_numpy()
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)
    return acc


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(
    pred_path: Path,
    sequence_dir: Path,
    fps: float,
    start_frame: int,
    end_frame: int,
    sample_seconds: float | None,
    summary_row: pd.Series,
) -> None:
    evaluated_seconds = (end_frame - start_frame + 1) / fps if fps > 0 else 0.0
    duration_note = (
        f"sample cap: {sample_seconds:.2f}s"
        if sample_seconds is not None
        else "sample cap: overlap only"
    )

    print()
    print("MOT Evaluation")
    print("=" * 70)
    print(f"Prediction file : {pred_path}")
    print(f"Sequence dir    : {sequence_dir}")
    print(f"Frame range     : {start_frame}..{end_frame} ({end_frame - start_frame + 1} frames)")
    print(f"Duration        : {evaluated_seconds:.2f}s at {fps:.2f} FPS ({duration_note})")
    print()
    print("Core Metrics")
    print("-" * 70)
    print(f"{'MOTA':<24}{pct(float(summary_row['mota']))}")
    print(f"{'MOTP':<24}{pct(float(summary_row['motp']))}")
    print(f"{'IDF1':<24}{pct(float(summary_row['idf1']))}")
    print(f"{'ID Precision':<24}{pct(float(summary_row['idp']))}")
    print(f"{'ID Recall':<24}{pct(float(summary_row['idr']))}")
    print()
    print("Tracking Counts")
    print("-" * 70)
    print(f"{'Frames evaluated':<24}{int(summary_row['num_frames'])}")
    print(f"{'GT objects':<24}{int(summary_row['num_objects'])}")
    print(f"{'False positives':<24}{int(summary_row['num_false_positives'])}")
    print(f"{'Misses':<24}{int(summary_row['num_misses'])}")
    print(f"{'ID switches':<24}{int(summary_row['num_switches'])}")
    print(f"{'Mostly tracked':<24}{int(summary_row['mostly_tracked'])}")
    print(f"{'Partially tracked':<24}{int(summary_row['partially_tracked'])}")
    print(f"{'Mostly lost':<24}{int(summary_row['mostly_lost'])}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    sequence_dir = Path(args.sequence_dir)
    gt_path = sequence_dir / "gt" / "gt.txt"
    seqinfo_path = sequence_dir / "seqinfo.ini"

    pred = load_prediction(pred_path)
    gt = load_ground_truth(gt_path)
    fps, seq_length = load_seqinfo(seqinfo_path)

    end_frame = determine_end_frame(pred, gt, seq_length, fps, args.sample_seconds)
    start_frame = max(1, int(args.start_frame))

    if end_frame < start_frame:
        raise ValueError(
            f"No overlapping frames to evaluate. start_frame={start_frame}, end_frame={end_frame}"
        )

    pred = pred[(pred["FrameId"] >= start_frame) & (pred["FrameId"] <= end_frame)].copy()
    gt = gt[(gt["FrameId"] >= start_frame) & (gt["FrameId"] <= end_frame)].copy()

    acc = accumulate_metrics(pred=pred, gt=gt, start_frame=start_frame, end_frame=end_frame)
    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(acc, metrics=SUMMARY_METRICS, name="eval")
    summary_row = summary.loc["eval"]

    print_report(
        pred_path=pred_path,
        sequence_dir=sequence_dir,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        sample_seconds=args.sample_seconds,
        summary_row=summary_row,
    )


if __name__ == "__main__":
    main()
