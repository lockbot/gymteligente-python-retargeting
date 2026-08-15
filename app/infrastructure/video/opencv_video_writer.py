"""Infrastructure: OpenCV implementation of the VideoWriter port."""
from __future__ import annotations

import cv2
import numpy as np

from app.domain.ports import VideoWriter


class OpenCvVideoWriter(VideoWriter):
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str = "mp4v",
    ):
        self._writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps or 30.0,
            (width, height),
        )
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()