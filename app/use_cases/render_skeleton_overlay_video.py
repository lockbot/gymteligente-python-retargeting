"""
Use case: Render Skeleton Overlay Video.

Re-reads the source video and, frame by frame, draws the corresponding
pose skeleton on top of it, writing the result to a new video file.
"""
from __future__ import annotations

from app.domain.entities import PoseSequence
from app.domain.ports import SkeletonOverlayRenderer, VideoReader, VideoWriter


class RenderSkeletonOverlayVideoUseCase:
    def __init__(
        self,
        video_reader: VideoReader,
        video_writer: VideoWriter,
        renderer: SkeletonOverlayRenderer,
    ):
        self._video_reader = video_reader
        self._video_writer = video_writer
        self._renderer = renderer

    def execute(self, pose_sequence: PoseSequence) -> None:
        for index, frame in enumerate(self._video_reader.frames()):
            pose_frame = pose_sequence.frames[index]
            rendered = self._renderer.render(frame, pose_frame)
            self._video_writer.write(rendered)
        self._video_writer.close()