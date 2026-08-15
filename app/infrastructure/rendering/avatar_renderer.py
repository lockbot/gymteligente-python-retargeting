"""
Infrastructure: draws a standalone skeletal avatar on a blank canvas,
driven purely by keypoint data -- i.e. motion retargeting / a
"digital twin" character reproducing the person's movements without
showing the original footage.

This is a lightweight 2D stick-avatar renderer. Swapping this class
for one that drives a 3D rigged model (e.g. via a game engine or a
physics-based ragdoll simulation) would not require touching the
domain or use-case layers, since both implement the same
`AvatarRenderer` port.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.domain.entities import PoseFrame, SKELETON_CONNECTIONS
from app.domain.ports import AvatarRenderer

BACKGROUND_COLOR = (30, 30, 30)   # dark gray, BGR
JOINT_COLOR = (255, 255, 255)     # white
BONE_COLOR = (60, 180, 255)       # orange, BGR
JOINT_RADIUS = 6
BONE_THICKNESS = 5
MIN_VISIBILITY_TO_DRAW = 0.5
NO_DETECTION_TEXT = "No pose detected"


class StickFigureAvatarRenderer(AvatarRenderer):
    def render(self, pose_frame: PoseFrame, width: int, height: int) -> np.ndarray:
        canvas = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)

        if not pose_frame.detected:
            cv2.putText(
                canvas,
                NO_DETECTION_TEXT,
                (int(width * 0.2), height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 120),
                2,
                lineType=cv2.LINE_AA,
            )
            return canvas

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
                canvas,
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
                canvas,
                kp.as_pixel(width, height),
                JOINT_RADIUS,
                JOINT_COLOR,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        return canvas