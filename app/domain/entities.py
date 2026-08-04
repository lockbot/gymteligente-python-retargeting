"""
Domain entities.

This module contains the core business objects of the pose-estimation
domain. It has ZERO dependencies on frameworks, ML libraries, or I/O —
pure Python + dataclasses only. This is what "Entities" means in Clean
Architecture: the innermost layer that everything else depends on, but
which depends on nothing else itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Landmark(str, Enum):
    """
    Canonical names for the 33 body landmarks tracked by the pose
    estimator (this mirrors the MediaPipe BlazePose topology, but the
    name is intentionally infrastructure-agnostic so a different
    pose-estimation engine could be swapped in without touching the
    domain).
    """
    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


# Ordered list matching the estimator's output index -> landmark name.
LANDMARK_ORDER: list[Landmark] = list(Landmark)

# Skeleton topology: pairs of landmark names that should be connected
# by a line when drawing a stick figure / avatar. Domain-level because
# "which joints connect to which" is a fact about human anatomy, not
# an implementation detail of any particular renderer.
SKELETON_CONNECTIONS: list[tuple[Landmark, Landmark]] = [
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.NOSE, Landmark.LEFT_SHOULDER),
    (Landmark.NOSE, Landmark.RIGHT_SHOULDER),
]


@dataclass(frozen=True)
class Keypoint:
    """A single tracked joint in a single frame, in normalized [0,1]
    image coordinates (origin top-left), plus a confidence score."""
    landmark: Landmark
    x: float
    y: float
    z: float
    visibility: float  # confidence / visibility score, 0.0 - 1.0

    def as_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to absolute pixel coordinates
        for a frame of the given size."""
        return int(self.x * width), int(self.y * height)


@dataclass(frozen=True)
class PoseFrame:
    """All detected keypoints for a single video frame. `keypoints` is
    None when no person/pose was detected in that frame."""
    frame_index: int
    timestamp_seconds: float
    keypoints: Optional[list[Keypoint]]

    @property
    def detected(self) -> bool:
        return self.keypoints is not None

    def get(self, landmark: Landmark) -> Optional[Keypoint]:
        if not self.keypoints:
            return None
        for kp in self.keypoints:
            if kp.landmark == landmark:
                return kp
        return None


@dataclass
class VideoInfo:
    """Metadata about a source or output video (this IS the technical
    sense of "metadata": data describing the video file, e.g. its
    resolution and frame rate -- as distinct from the pose keypoint
    data itself)."""
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class PoseSequence:
    """The full pose-estimation result for a video: one PoseFrame per
    frame, plus the source video's technical metadata."""
    source_video_info: VideoInfo
    frames: list[PoseFrame] = field(default_factory=list)

    @property
    def detected_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.detected)
