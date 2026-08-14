"""
Use case: Process Video (orchestrator).

Coordinates the full pipeline for one input video:
  1. Extract pose keypoint data for every frame.
  2. Persist that keypoint data to structured storage.
  3. Render the skeleton-overlay video.
  4. Render the standalone avatar (motion-retargeting) video.

This is an "application service" -- it composes the smaller use cases
above. It knows nothing about FastAPI, MediaPipe, or OpenCV; those are
injected as ports by the composition root.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities import PoseSequence
from app.domain.ports import (
    AvatarRenderer,
    PoseEstimator,
    PoseSequenceStorage,
    SkeletonOverlayRenderer,
    VideoReader,
    VideoWriter,
)
from app.use_cases.extract_pose_sequence import ExtractPoseSequenceUseCase
from app.use_cases.render_avatar_video import RenderAvatarVideoUseCase
from app.use_cases.render_skeleton_overlay_video import RenderSkeletonOverlayVideoUseCase
from app.use_cases.save_pose_data import SavePoseDataUseCase


@dataclass
class ProcessVideoResult:
    pose_sequence: PoseSequence
    pose_data_path: str
    skeleton_overlay_video_path: str
    avatar_video_path: str


class ProcessVideoUseCase:
    """
    Note the constructor takes *factories* (callables) for the video
    readers/writers rather than already-open instances, because the
    source video needs to be read twice (once for pose extraction,
    once for overlay rendering) and each output video needs its own
    writer. Factories let the composition root control exactly how/
    where those readers and writers are created without this use case
    knowing about file paths or codecs.
    """

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        pose_storage: PoseSequenceStorage,
        skeleton_renderer: SkeletonOverlayRenderer,
        avatar_renderer: AvatarRenderer,
        make_source_reader: "callable[[], VideoReader]",
        make_overlay_writer: "callable[[], VideoWriter]",
        make_avatar_writer: "callable[[], VideoWriter]",
    ):
        self._pose_estimator = pose_estimator
        self._pose_storage = pose_storage
        self._skeleton_renderer = skeleton_renderer
        self._avatar_renderer = avatar_renderer
        self._make_source_reader = make_source_reader
        self._make_overlay_writer = make_overlay_writer
        self._make_avatar_writer = make_avatar_writer

    def execute(
        self,
        pose_data_output_path: str,
        overlay_video_output_path: str,
        avatar_video_output_path: str,
    ) -> ProcessVideoResult:
        # 1. Extract pose keypoints for every frame.
        reader_for_extraction = self._make_source_reader()
        extract_use_case = ExtractPoseSequenceUseCase(
            pose_estimator=self._pose_estimator,
            video_reader=reader_for_extraction,
        )
        pose_sequence = extract_use_case.execute()
        reader_for_extraction.close()

        # 2. Persist the keypoint data.
        save_use_case = SavePoseDataUseCase(storage=self._pose_storage)
        pose_data_path = save_use_case.execute(pose_sequence, pose_data_output_path)

        # 3. Render the skeleton-overlay video (re-reads source video).
        reader_for_overlay = self._make_source_reader()
        overlay_use_case = RenderSkeletonOverlayVideoUseCase(
            video_reader=reader_for_overlay,
            video_writer=self._make_overlay_writer(),
            renderer=self._skeleton_renderer,
        )
        overlay_use_case.execute(pose_sequence)
        reader_for_overlay.close()

        # 4. Render the standalone avatar video.
        avatar_use_case = RenderAvatarVideoUseCase(
            video_writer=self._make_avatar_writer(),
            renderer=self._avatar_renderer,
        )
        avatar_use_case.execute(pose_sequence)

        return ProcessVideoResult(
            pose_sequence=pose_sequence,
            pose_data_path=pose_data_path,
            skeleton_overlay_video_path=overlay_video_output_path,
            avatar_video_path=avatar_video_output_path,
        )