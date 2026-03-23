from __future__ import annotations

from pathlib import Path

from src.schemas import Track


class MOTExporter:
    def export(self, tracks: list[Track], output_path: str) -> str:
        mot_path = str(Path(output_path).with_suffix(".mot.txt"))
        Path(mot_path).parent.mkdir(parents=True, exist_ok=True)

        latest_per_key: dict[tuple[int, int], Track] = {}
        for track in tracks:
            latest_per_key[(track.frame_index, track.track_id)] = track

        sorted_tracks = sorted(
            latest_per_key.values(),
            key=lambda track: (track.frame_index, track.track_id),
        )
        lines = [self._format_track(track) for track in sorted_tracks]

        with Path(mot_path).open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        return mot_path

    def _format_track(self, track: Track) -> str:
        x1, y1, x2, y2 = track.bbox
        width = x2 - x1
        height = y2 - y1
        mot_frame = track.frame_index + 1
        return (
            f"{mot_frame},{track.track_id},"
            f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
            f"{track.confidence:.4f},-1,-1,-1"
        )
