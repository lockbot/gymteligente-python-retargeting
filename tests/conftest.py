"""
Pytest fixtures: synthetic test video generation
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FRAME_WIDTH = 160
FRAME_HEIGHT = 120
FRAME_COUNT = 10
FPS = 10.0


@pytest.fixture()
def synthetic_video_path(tmp_path: Path) -> str:
    """Generate a tiny synthetic .mp4 (a moving colored square on a
    black background) so tests don't depend on any real footage."""
    video_path = tmp_path / "synthetic_input.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    for i in range(FRAME_COUNT):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        x = 10 + i * 5
        cv2.rectangle(frame, (x, 40), (x + 20, 80), (0, 0, 255), thickness=-1)
        writer.write(frame)
    writer.release()
    return str(video_path)