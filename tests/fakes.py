"""
Test double for PoseEstimator port, used across the test suite.

Lets the whole pipeline (extraction -> rendering -> storage -> API) be
tested without needing the real MediaPipe model file, which isn't
downloaded automatically (see scripts/download_pose_model.py). This is
exactly the kind of substitution the ports/adapters split in
`domain/ports.py` is meant to enable.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from app.domain.entities import Keypoint, LANDMARK_ORDER
from app.domain.ports import PoseEstimator


class FakePoseEstimator(PoseEstimator):
    """Returns deterministic, synthetic keypoints for every frame
    except frames whose index is in `skip_frame_indices` (used to
    simulate 'no person detected' frames)."""

    def __init__(self, skip_frame_indices: frozenset[int] = frozenset()):
        self._skip_frame_indices = skip_frame_indices
        self._call_count = 0

    def estimate(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Optional[list[Keypoint]]:
        current_call = self._call_count
        self._call_count += 1

        if current_call in self._skip_frame_indices:
            return None

        # Cheap deterministic "motion": every landmark orbits slightly
        # based on the call index, just so frames aren't identical.
        phase = current_call * 0.1
        keypoints = []
        for i, landmark in enumerate(LANDMARK_ORDER):
            x = 0.5 + 0.05 * math.sin(phase + i)
            y = 0.5 + 0.05 * math.cos(phase + i)
            keypoints.append(
                Keypoint(landmark=landmark, x=x, y=y, z=0.0, visibility=0.9)
            )
        return keypoints