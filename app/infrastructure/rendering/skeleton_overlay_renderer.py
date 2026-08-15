"""
Infrastructure: draws the keypoint skeleton on top of the original
video frame -- the classic "green stick figure over the footage" look.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.domain.entities import PoseFrame, SKELETON_CONNECTIONS
from app.domain.ports import SkeletonOverlayRenderer

JOINT_COLOR = (0, 255, 0)   # green, BGR
BONE_COLOR = (0, 200, 0)    # slightly darker green, BGR
JOINT_RADIUS = 4
BONE_THICKNESS = 2
MIN_VISIBILITY_TO_DRAW = 0.5


class OpenCvSkeletonOverlayRenderer(SkeletonOverlayRenderer):
    def render(self, frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        output = frame.copy()
        if not pose_frame.detected:
            return output

        height, width = output.shape[:2]

        for start_landmark, end_landmark in SKELETON_CONNECTIONS:
            start_kp = pose_frame.get(start_landmark)
            end_kp = pose_frame.get(end_landmark)
            if not start_kp or not end_kp:
                continue
            if (
                start_kp.visibility < MIN_VISIBILITY_TO_DRAW
                or end_kp.visibility < MIN_VISIBILITY_TO_DRAW
            ):
                continue
            cv2.line(
                output,
                start_kp.as_pixel(width, height),
                end_kp.as_pixel(width, height),
                BONE_COLOR,
                BONE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        for kp in pose_frame.keypoints:
            if kp.visibility < MIN_VISIBILITY_TO_DRAW:
                continue
            cv2.circle(
                output,
                kp.as_pixel(width, height),
                JOINT_RADIUS,
                JOINT_COLOR,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        return output