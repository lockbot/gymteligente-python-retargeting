"""
Use case: Render Avatar Video.

Generates a standalone video of a skeletal avatar reproducing the
person's movements, driven entirely by the extracted keypoint data
(no original footage involved). This is the "motion retargeting /
ragdoll-style" deliverable.
"""
from __future__ import annotations

from app.domain.entities import PoseSequence
from app.domain.ports import AvatarRenderer, VideoWriter


class RenderAvatarVideoUseCase:
    def __init__(self, video_writer: VideoWriter, renderer: AvatarRenderer):
        self._video_writer = video_writer
        self._renderer = renderer

    def execute(self, pose_sequence: PoseSequence) -> None:
        width = pose_sequence.source_video_info.width
        height = pose_sequence.source_video_info.height
        for pose_frame in pose_sequence.frames:
            rendered = self._renderer.render(pose_frame, width, height)
            self._video_writer.write(rendered)
        self._video_writer.close()