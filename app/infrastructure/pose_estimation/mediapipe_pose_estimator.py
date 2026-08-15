"""
Infrastructure: MediaPipe implementation of the PoseEstimator port.

This is the only file in the project that imports `mediapipe`. If we
ever swap in a different engine (YOLO-Pose, MMPose, OpenPose...), we
add a new class here implementing the same `PoseEstimator` port --
nothing in `domain/` or `use_cases/` has to change.
"""
from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from app.domain.entities import Keypoint, LANDMARK_ORDER
from app.domain.ports import PoseEstimator


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )

    def estimate(self, frame: np.ndarray) -> Optional[list[Keypoint]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        keypoints: list[Keypoint] = []
        for landmark_enum, mp_landmark in zip(
            LANDMARK_ORDER, results.pose_landmarks.landmark
        ):
            keypoints.append(
                Keypoint(
                    landmark=landmark_enum,
                    x=mp_landmark.x,
                    y=mp_landmark.y,
                    z=mp_landmark.z,
                    visibility=mp_landmark.visibility,
                )
            )
        return keypoints

    def close(self) -> None:
        self._pose.close()