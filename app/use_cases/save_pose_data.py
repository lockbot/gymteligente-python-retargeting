"""
Use case: Save Pose Data.

Persists the extracted PoseSequence to structured storage (JSON/CSV/...)
via the PoseSequenceStorage port.
"""
from __future__ import annotations

from app.domain.entities import PoseSequence
from app.domain.ports import PoseSequenceStorage


class SavePoseDataUseCase:
    def __init__(self, storage: PoseSequenceStorage):
        self._storage = storage

    def execute(self, pose_sequence: PoseSequence, destination_path: str) -> str:
        return self._storage.save(pose_sequence, destination_path)