"""
Composition root.

This is the ONE place in the whole codebase allowed to know about both
the abstract ports (domain) and their concrete infrastructure
implementations at the same time, and to wire them together. Routes
call `build_process_video_use_case(...)`; they never instantiate
MediaPipe/OpenCV classes directly.
"""
from __future__ import annotations

from app.domain.ports import PoseSequenceStorage
from app.infrastructure.pose_estimation.mediapipe_pose_estimator import (
    MediaPipePoseEstimator,
)
from app.infrastructure.rendering.avatar_renderer import StickFigureAvatarRenderer
from app.infrastructure.rendering.skeleton_overlay_renderer import (
    OpenCvSkeletonOverlayRenderer,
)
from app.infrastructure.storage.csv_pose_storage import CsvPoseSequenceStorage
from app.infrastructure.storage.json_pose_storage import JsonPoseSequenceStorage
from app.infrastructure.video.opencv_video_reader import OpenCvVideoReader
from app.infrastructure.video.opencv_video_writer import OpenCvVideoWriter
from app.use_cases.process_video import ProcessVideoUseCase


def _build_pose_storage(pose_data_format: str) -> PoseSequenceStorage:
    if pose_data_format == "csv":
        return CsvPoseSequenceStorage()
    return JsonPoseSequenceStorage()


def build_process_video_use_case(
    source_video_path: str,
    overlay_output_path: str,
    avatar_output_path: str,
    pose_data_format: str = "json",
) -> ProcessVideoUseCase:
    """
    Build a fully-wired ProcessVideoUseCase for one job. A fresh
    PoseEstimator is created per job to keep MediaPipe's internal
    (stateful, non-thread-safe) tracker isolated between requests.
    """
    pose_estimator = MediaPipePoseEstimator()
    pose_storage = _build_pose_storage(pose_data_format)
    skeleton_renderer = OpenCvSkeletonOverlayRenderer()
    avatar_renderer = StickFigureAvatarRenderer()

    # We need video dimensions/fps to open the output writers, so we
    # peek at the source video's metadata up front via a throwaway
    # reader.
    probe_reader = OpenCvVideoReader(source_video_path)
    info = probe_reader.get_info()
    probe_reader.close()

    return ProcessVideoUseCase(
        pose_estimator=pose_estimator,
        pose_storage=pose_storage,
        skeleton_renderer=skeleton_renderer,
        avatar_renderer=avatar_renderer,
        make_source_reader=lambda: OpenCvVideoReader(source_video_path),
        make_overlay_writer=lambda: OpenCvVideoWriter(
            overlay_output_path, info.width, info.height, info.fps
        ),
        make_avatar_writer=lambda: OpenCvVideoWriter(
            avatar_output_path, info.width, info.height, info.fps
        ),
    )