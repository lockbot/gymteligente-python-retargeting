"""Infrastructure: JSON implementation of the PoseSequenceStorage port."""
from __future__ import annotations

import json
import os

from app.domain.entities import PoseSequence
from app.domain.ports import PoseSequenceStorage


class JsonPoseSequenceStorage(PoseSequenceStorage):
    def save(self, pose_sequence: PoseSequence, destination_path: str) -> str:
        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

        payload = {
            "video_info": {
                "width": pose_sequence.source_video_info.width,
                "height": pose_sequence.source_video_info.height,
                "fps": pose_sequence.source_video_info.fps,
                "frame_count": pose_sequence.source_video_info.frame_count,
            },
            "frame_count": len(pose_sequence.frames),
            "detected_frame_count": pose_sequence.detected_frame_count,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "detected": frame.detected,
                    "keypoints": (
                        [
                            {
                                "landmark": kp.landmark.value,
                                "x": kp.x,
                                "y": kp.y,
                                "z": kp.z,
                                "visibility": kp.visibility,
                            }
                            for kp in frame.keypoints
                        ]
                        if frame.keypoints
                        else []
                    ),
                }
                for frame in pose_sequence.frames
            ],
        }

        with open(destination_path, "w") as f:
            json.dump(payload, f, indent=2)

        return destination_path