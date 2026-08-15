"""
Interface adapters: FastAPI routes.

Thin HTTP controllers. They handle upload/download plumbing and job
bookkeeping, then delegate all real work to the use-case layer via the
composition root in `container.py`. No business logic lives here.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.interface_adapters.api.container import build_process_video_use_case
from app.interface_adapters.api.schemas import HealthResponse, ProcessVideoResponse

router = APIRouter()

# NOTE: in-memory job registry for demo purposes. Swap for a database
# or job queue (Celery/RQ) in production -- the use-case layer already
# doesn't care how/where this bookkeeping happens.
_JOBS: dict[str, dict[str, str]] = {}

UPLOADS_DIR = settings.uploads_dir
OUTPUTS_DIR = settings.outputs_dir
for _dir in (UPLOADS_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()


@router.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(
    file: UploadFile = File(..., description="Video of a person performing an action."),
    pose_data_format: Literal["json", "csv"] = Query(default="json", pattern="^(json|csv)$"),
) -> ProcessVideoResponse:
    job_id = str(uuid.uuid4())
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    source_video_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with open(source_video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pose_data_path = job_dir / f"pose_data.{pose_data_format}"
    overlay_video_path = job_dir / "skeleton_overlay.mp4"
    avatar_video_path = job_dir / "avatar.mp4"

    try:
        use_case = build_process_video_use_case(
            source_video_path=str(source_video_path),
            overlay_output_path=str(overlay_video_path),
            avatar_output_path=str(avatar_video_path),
            pose_data_format=pose_data_format,
        )
        result = use_case.execute(
            pose_data_output_path=str(pose_data_path),
            overlay_video_output_path=str(overlay_video_path),
            avatar_video_output_path=str(avatar_video_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _JOBS[job_id] = {
        "pose_data": str(pose_data_path),
        "overlay_video": str(overlay_video_path),
        "avatar_video": str(avatar_video_path),
    }

    return ProcessVideoResponse(
        job_id=job_id,
        frame_count=len(result.pose_sequence.frames),
        detected_frame_count=result.pose_sequence.detected_frame_count,
        pose_data_format=pose_data_format,
        pose_data_download_url=f"/jobs/{job_id}/download/pose-data",
        skeleton_overlay_video_download_url=f"/jobs/{job_id}/download/skeleton-overlay",
        avatar_video_download_url=f"/jobs/{job_id}/download/avatar",
    )


_ARTIFACT_KEYS = {
    "pose-data": "pose_data",
    "skeleton-overlay": "overlay_video",
    "avatar": "avatar_video",
}


@router.get("/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str) -> FileResponse:
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    if artifact not in _ARTIFACT_KEYS:
        raise HTTPException(status_code=404, detail="Unknown artifact type")

    path = _JOBS[job_id][_ARTIFACT_KEYS[artifact]]
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(path)