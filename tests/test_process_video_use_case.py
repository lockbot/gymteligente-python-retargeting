"""
Test: full ProcessVideoUseCase orchestration
"""

from __future__ import annotations

from pathlib import Path

from app.infrastructure.rendering.avatar_renderer import StickFigureAvatarRenderer
from app.infrastructure.rendering.skeleton_overlay_renderer import (
    OpenCvSkeletonOverlayRenderer,
)
from app.infrastructure.storage.json_pose_storage import JsonPoseSequenceStorage
from app.infrastructure.video.opencv_video_reader import OpenCvVideoReader
from app.infrastructure.video.opencv_video_writer import OpenCvVideoWriter
from app.use_cases.process_video import ProcessVideoUseCase
from tests.conftest import FRAME_COUNT
from tests.fakes import FakePoseEstimator


def test_process_video_use_case_produces_all_three_artifacts(
    synthetic_video_path, tmp_path: Path
):
    probe = OpenCvVideoReader(synthetic_video_path)
    info = probe.get_info()
    probe.close()

    pose_data_path = tmp_path / "pose_data.json"
    overlay_path = tmp_path / "overlay.mp4"
    avatar_path = tmp_path / "avatar.mp4"

    use_case = ProcessVideoUseCase(
        pose_estimator=FakePoseEstimator(),
        pose_storage=JsonPoseSequenceStorage(),
        skeleton_renderer=OpenCvSkeletonOverlayRenderer(),
        avatar_renderer=StickFigureAvatarRenderer(),
        make_source_reader=lambda: OpenCvVideoReader(synthetic_video_path),
        make_overlay_writer=lambda: OpenCvVideoWriter(
            str(overlay_path), info.width, info.height, info.fps
        ),
        make_avatar_writer=lambda: OpenCvVideoWriter(
            str(avatar_path), info.width, info.height, info.fps
        ),
    )

    result = use_case.execute(
        pose_data_output_path=str(pose_data_path),
        overlay_video_output_path=str(overlay_path),
        avatar_video_output_path=str(avatar_path),
    )

    assert len(result.pose_sequence.frames) == FRAME_COUNT
    assert pose_data_path.exists()
    assert overlay_path.exists()
    assert avatar_path.exists()