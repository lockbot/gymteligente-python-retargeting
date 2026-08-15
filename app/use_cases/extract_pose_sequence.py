"""
Use case: Extract Pose Sequence.

Reads every frame of a source video and runs pose estimation on it,
producing a PoseSequence domain object. Depends only on the
PoseEstimator and VideoReader ports -- never on concrete
MediaPipe/OpenCV classes.
"""
from __future__ import annotations

from app.domain.entities import PoseFrame, PoseSequence
from app.domain.ports import PoseEstimator, VideoReader


class ExtractPoseSequenceUseCase:
    def __init__(self, pose_estimator: PoseEstimator, video_reader: VideoReader):
        self._pose_estimator = pose_estimator
        self._video_reader = video_reader

    def execute(self) -> PoseSequence:
        video_info = self._video_reader.get_info()
        fps = video_info.fps or 30.0

        pose_frames: list[PoseFrame] = []
        for index, frame in enumerate(self._video_reader.frames()):
            timestamp_seconds = index / fps
            timestamp_ms = int(timestamp_seconds * 1000)
            keypoints = self._pose_estimator.estimate(frame, timestamp_ms)
            pose_frames.append(
                PoseFrame(
                    frame_index=index,
                    timestamp_seconds=timestamp_seconds,
                    keypoints=keypoints,
                )
            )

        return PoseSequence(source_video_info=video_info, frames=pose_frames)