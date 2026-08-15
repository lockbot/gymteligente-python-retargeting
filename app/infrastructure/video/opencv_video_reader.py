"""Infrastructure: OpenCV implementation of the VideoReader port."""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from app.domain.entities import VideoInfo
from app.domain.ports import VideoReader


class OpenCvVideoReader(VideoReader):
    def __init__(self, video_path: str):
        self._video_path = video_path
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

    def get_info(self) -> VideoInfo:
        return VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[np.ndarray]:
        # Rewind so this reader can be iterated more than once if the
        # caller wishes; the pipeline itself opens a fresh reader per
        # pass, but this keeps the class self-contained and reusable.
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

    def close(self) -> None:
        self._capture.release()