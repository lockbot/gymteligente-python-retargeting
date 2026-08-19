"""
Test: FastAPI routes end-to-end via TestClient, with the real MediaPipe estimator swapped for the fake
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.interface_adapters.api import container as container_module
from tests.conftest import FRAME_COUNT
from tests.fakes import FakePoseEstimator


@pytest.fixture()
def client(monkeypatch, tmp_path: Path) -> TestClient:
    # Swap the real MediaPipe estimator for the fake so this test
    # doesn't need the downloaded .task model file -- everything else
    # (upload handling, use-case wiring, rendering, storage, download
    # endpoints) runs for real.
    monkeypatch.setattr(
        container_module,
        "MediaPipePoseEstimator",
        lambda *args, **kwargs: FakePoseEstimator(),
    )
    monkeypatch.setattr(container_module.settings, "data_root", tmp_path / "data")

    import app.interface_adapters.api.routes as routes_module

    monkeypatch.setattr(routes_module, "UPLOADS_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(routes_module, "OUTPUTS_DIR", tmp_path / "data" / "outputs")
    routes_module.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    routes_module.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    from main import app

    return TestClient(app)


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_video_end_to_end(client: TestClient, synthetic_video_path: str):
    with open(synthetic_video_path, "rb") as video_file:
        response = client.post(
            "/process-video",
            files={"file": ("exercise.mp4", video_file, "video/mp4")},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_count"] == FRAME_COUNT
    assert body["detected_frame_count"] == FRAME_COUNT
    assert body["pose_data_format"] == "json"

    job_id = body["job_id"]

    pose_data_response = client.get(f"/jobs/{job_id}/download/pose-data")
    assert pose_data_response.status_code == 200

    overlay_response = client.get(f"/jobs/{job_id}/download/skeleton-overlay")
    assert overlay_response.status_code == 200

    avatar_response = client.get(f"/jobs/{job_id}/download/avatar")
    assert avatar_response.status_code == 200