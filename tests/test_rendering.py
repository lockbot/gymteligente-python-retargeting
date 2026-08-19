"""
Test: skeleton overlay and avatar video rendering use cases
"""

from __future__ import annotations

from pathlib import Path

import cv2

from app.infrastructure.rendering.avatar_renderer import StickFigureAvatarRenderer
from app.infrastructure.rendering.skeleton_overlay_renderer import (
    OpenCvSkeletonOverlayRenderer,
)
from app.infrastructure.video.opencv_video_reader import OpenCvVideoReader
from app.infrastructure.video.opencv_video_writer import OpenCvVideoWriter
from app.use_cases.extract_pose_sequence import ExtractPoseSequenceUseCase
from app.use_cases.render_avatar_video import RenderAvatarVideoUseCase
from app.use_cases.render_skeleton_overlay_video import (
    RenderSkeletonOverlayVideoUseCase,
)
from tests.conftest import FRAME_COUNT
from tests.fakes import FakePoseEstimator


def _extract(video_path: str):
    reader = OpenCvVideoReader(video_path)
    use_case = ExtractPoseSequenceUseCase(
        pose_estimator=FakePoseEstimator(), video_reader=reader
    )
    return use_case.execute()


def test_render_skeleton_overlay_video_writes_expected_frame_count(
    synthetic_video_path, tmp_path: Path
):
    pose_sequence = _extract(synthetic_video_path)
    output_path = tmp_path / "overlay.mp4"

    use_case = RenderSkeletonOverlayVideoUseCase(
        video_reader=OpenCvVideoReader(synthetic_video_path),
        video_writer=OpenCvVideoWriter(
            str(output_path),
            pose_sequence.source_video_info.width,
            pose_sequence.source_video_info.height,
            pose_sequence.source_video_info.fps,
        ),
        renderer=OpenCvSkeletonOverlayRenderer(),
    )
    use_case.execute(pose_sequence)

    assert output_path.exists()
    capture = cv2.VideoCapture(str(output_path))
    written_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert written_frame_count == FRAME_COUNT


def test_render_avatar_video_writes_expected_frame_count(
    synthetic_video_path, tmp_path: Path
):
    pose_sequence = _extract(synthetic_video_path)
    output_path = tmp_path / "avatar.mp4"

    use_case = RenderAvatarVideoUseCase(
        video_writer=OpenCvVideoWriter(
            str(output_path),
            pose_sequence.source_video_info.width,
            pose_sequence.source_video_info.height,
            pose_sequence.source_video_info.fps,
        ),
        renderer=StickFigureAvatarRenderer(),
    )
    use_case.execute(pose_sequence)

    assert output_path.exists()
    capture = cv2.VideoCapture(str(output_path))
    written_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert written_frame_count == FRAME_COUNT