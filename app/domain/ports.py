"""
Ports (a.k.a. repository interfaces / gateways).

These are abstract contracts that the domain and use-case layers
depend on. Concrete implementations live in `infrastructure/` and are
"plugged in" at the composition root (see
`interface_adapters/api/container.py`). This is the Dependency
Inversion piece of Clean Architecture: inner layers define the
interface, outer layers implement it -- never the reverse.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np

from app.domain.entities import Keypoint, PoseFrame, PoseSequence, VideoInfo


class PoseEstimator(ABC):
    """Port for any engine that can detect body keypoints in a frame."""

    @abstractmethod
    def estimate(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Optional[list[Keypoint]]:
        """Return detected keypoints for a single BGR image frame, or
        None if no pose was detected.

        `timestamp_ms` is the frame's position in the source video in
        milliseconds and MUST be strictly increasing across successive
        calls for the same estimator instance -- video-mode pose
        trackers use it to maintain temporal continuity between
        frames."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook (release model resources etc.)."""
        return None


class VideoReader(ABC):
    """Port for reading frames + metadata from a source video."""

    @abstractmethod
    def get_info(self) -> VideoInfo:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames in order as BGR numpy arrays."""
        raise NotImplementedError

    def close(self) -> None:
        return None


class VideoWriter(ABC):
    """Port for writing frames out to a video file."""

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class SkeletonOverlayRenderer(ABC):
    """Port for drawing the keypoint skeleton on top of an existing
    video frame (the 'green stick figure over the footage' output)."""

    @abstractmethod
    def render(self, frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        """Return a new frame with the skeleton drawn over `frame`."""
        raise NotImplementedError


class AvatarRenderer(ABC):
    """Port for drawing a standalone skeletal avatar (stick figure /
    ragdoll-style character) driven purely by keypoint data, on a
    blank canvas -- this is the 'motion retargeting' output."""

    @abstractmethod
    def render(self, pose_frame: PoseFrame, width: int, height: int) -> np.ndarray:
        """Return a new blank-canvas frame with the avatar drawn in
        the pose described by `pose_frame`."""
        raise NotImplementedError


class PoseSequenceStorage(ABC):
    """Port for persisting a PoseSequence to structured storage
    (JSON, CSV, database, ...)."""

    @abstractmethod
    def save(self, pose_sequence: PoseSequence, destination_path: str) -> str:
        """Persist the pose sequence and return the path/URI it was
        written to."""
        raise NotImplementedError