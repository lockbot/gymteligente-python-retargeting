"""
Application configuration.

Plain-dataclass settings sourced from environment variables. Kept
dependency-free (no pydantic-settings) since the project's
pyproject.toml doesn't declare it -- add it later if config needs grow
past this.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # -- Pose estimation model --
    pose_model_path: str = os.environ.get(
        "POSE_MODEL_PATH", str(PROJECT_ROOT / "models" / "pose_landmarker_lite.task")
    )
    min_detection_confidence: float = float(
        os.environ.get("POSE_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    min_tracking_confidence: float = float(
        os.environ.get("POSE_MIN_TRACKING_CONFIDENCE", "0.5")
    )

    # -- Storage --
    data_root: Path = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))

    # -- Server --
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    reload: bool = os.environ.get("RELOAD", "false").lower() == "true"

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_root / "outputs"


settings = Settings()
