"""Infrastructure: CSV implementation of the PoseSequenceStorage port.

One row per (frame, landmark) pair -- a "long" / tidy format that's
easy to load into pandas for downstream analysis (e.g. joint-angle
calculations for exercise form checking).
"""
from __future__ import annotations

import csv
import os

from app.domain.entities import PoseSequence
from app.domain.ports import PoseSequenceStorage

FIELDNAMES = [
    "frame_index",
    "timestamp_seconds",
    "detected",
    "landmark",
    "x",
    "y",
    "z",
    "visibility",
]


class CsvPoseSequenceStorage(PoseSequenceStorage):
    def save(self, pose_sequence: PoseSequence, destination_path: str) -> str:
        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

        with open(destination_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            for frame in pose_sequence.frames:
                if not frame.keypoints:
                    writer.writerow(
                        {
                            "frame_index": frame.frame_index,
                            "timestamp_seconds": frame.timestamp_seconds,
                            "detected": frame.detected,
                            "landmark": "",
                            "x": "",
                            "y": "",
                            "z": "",
                            "visibility": "",
                        }
                    )
                    continue

                for kp in frame.keypoints:
                    writer.writerow(
                        {
                            "frame_index": frame.frame_index,
                            "timestamp_seconds": frame.timestamp_seconds,
                            "detected": frame.detected,
                            "landmark": kp.landmark.value,
                            "x": kp.x,
                            "y": kp.y,
                            "z": kp.z,
                            "visibility": kp.visibility,
                        }
                    )

        return destination_path