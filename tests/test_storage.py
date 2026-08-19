"""
Test: JSON and CSV pose data storage
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.infrastructure.storage.csv_pose_storage import CsvPoseSequenceStorage
from app.infrastructure.storage.json_pose_storage import JsonPoseSequenceStorage
from app.infrastructure.video.opencv_video_reader import OpenCvVideoReader
from app.use_cases.extract_pose_sequence import ExtractPoseSequenceUseCase
from tests.conftest import FRAME_COUNT
from tests.fakes import FakePoseEstimator


def _extract(video_path: str):
    reader = OpenCvVideoReader(video_path)
    use_case = ExtractPoseSequenceUseCase(
        pose_estimator=FakePoseEstimator(), video_reader=reader
    )
    return use_case.execute()


def test_json_storage_round_trip(synthetic_video_path, tmp_path: Path):
    pose_sequence = _extract(synthetic_video_path)
    destination = tmp_path / "pose_data.json"

    CsvJsonPathResult = JsonPoseSequenceStorage().save(pose_sequence, str(destination))

    assert CsvJsonPathResult == str(destination)
    payload = json.loads(destination.read_text())
    assert payload["frame_count"] == FRAME_COUNT
    assert len(payload["frames"]) == FRAME_COUNT
    assert len(payload["frames"][0]["keypoints"]) == 33  # 33 BlazePose landmarks


def test_csv_storage_round_trip(synthetic_video_path, tmp_path: Path):
    pose_sequence = _extract(synthetic_video_path)
    destination = tmp_path / "pose_data.csv"

    CsvPoseSequenceStorage().save(pose_sequence, str(destination))

    with open(destination) as f:
        rows = list(csv.DictReader(f))

    # 33 landmark rows per frame, all frames detected.
    assert len(rows) == FRAME_COUNT * 33
    assert rows[0]["landmark"] == "nose"