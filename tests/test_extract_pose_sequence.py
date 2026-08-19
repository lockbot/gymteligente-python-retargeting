"""
Test: ExtractPoseSequenceUseCase against a synthetic video
"""

from __future__ import annotations

from app.infrastructure.video.opencv_video_reader import OpenCvVideoReader
from app.use_cases.extract_pose_sequence import ExtractPoseSequenceUseCase
from tests.conftest import FRAME_COUNT
from tests.fakes import FakePoseEstimator


def test_extract_pose_sequence_detects_every_frame(synthetic_video_path):
    reader = OpenCvVideoReader(synthetic_video_path)
    use_case = ExtractPoseSequenceUseCase(
        pose_estimator=FakePoseEstimator(), video_reader=reader
    )

    pose_sequence = use_case.execute()

    assert len(pose_sequence.frames) == FRAME_COUNT
    assert pose_sequence.detected_frame_count == FRAME_COUNT
    assert pose_sequence.source_video_info.width == 160
    assert pose_sequence.source_video_info.height == 120


def test_extract_pose_sequence_handles_missed_detections(synthetic_video_path):
    reader = OpenCvVideoReader(synthetic_video_path)
    use_case = ExtractPoseSequenceUseCase(
        pose_estimator=FakePoseEstimator(skip_frame_indices=frozenset({0, 3})),
        video_reader=reader,
    )

    pose_sequence = use_case.execute()

    assert pose_sequence.detected_frame_count == FRAME_COUNT - 2
    assert pose_sequence.frames[0].detected is False
    assert pose_sequence.frames[3].detected is False
    assert pose_sequence.frames[1].detected is True