"""
Interface adapters: Pydantic schemas for the HTTP API boundary.

These convert between the wire format (JSON over HTTP) and domain
objects. They live in the "interface adapters" layer -- domain and
use-case code never imports from this module.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProcessVideoResponse(BaseModel):
    job_id: str
    frame_count: int
    detected_frame_count: int
    pose_data_format: Literal["json", "csv"]
    pose_data_download_url: str
    skeleton_overlay_video_download_url: str
    avatar_video_download_url: str


class HealthResponse(BaseModel):
    status: str = "ok"


class ProcessVideoOptions(BaseModel):
    pose_data_format: Literal["json", "csv"] = Field(
        default="json",
        description="Structured format to save the extracted keypoint data in.",
    )