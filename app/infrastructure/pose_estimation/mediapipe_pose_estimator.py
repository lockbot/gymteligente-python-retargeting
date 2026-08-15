"""
Infrastructure: MediaPipe implementation of the PoseEstimator port.

Built against MediaPipe's Tasks API (`mediapipe.tasks.python.vision.
PoseLandmarker`), which is the current API as of mediapipe>=1.0. The
older `mp.solutions.pose.Pose` API used in mediapipe 0.10.x has been
removed from the package entirely in 1.0+.

This is the only file in the project that imports `mediapipe`. If we
ever swap in a different engine (YOLO-Pose, MMPose, OpenPose...), we
add a new class here implementing the same `PoseEstimator` port --
nothing in `domain/` or `use_cases/` has to change.

IMPORTANT: this class requires a `.task` model bundle file on disk
(see `app/config.py` for the default path, and
`scripts/download_pose_model.py` to fetch one). MediaPipe does not
ship the model inside the pip package.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app.domain.entities import Keypoint, LANDMARK_ORDER
from app.domain.ports import PoseEstimator


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{model_path}'. "
                "Run `python scripts/download_pose_model.py` first, or "
                "point POSE_MODEL_PATH at an existing .task file. "
                "See README.md > 'Download the pose model' for details."
            )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(
        self, frame: np.ndarray, timestamp_ms: int
        ) -> Optional[list[Keypoint]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        # num_poses=1, so we only ever care about the first detected person.
        raw_landmarks = result.pose_landmarks[0]

        keypoints: list[Keypoint] = []
        for landmark_enum, raw in zip(LANDMARK_ORDER, raw_landmarks):
            keypoints.append(
                Keypoint(
                    landmark=landmark_enum,
                    x=raw.x,
                    y=raw.y,
                    z=raw.z or 0.0,
                    visibility=raw.visibility if raw.visibility is not None else 0.0,
                )
            )
        return keypoints

    def close(self) -> None:
        self._landmarker.close()
