"""Interface adapters: FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from app.interface_adapters.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Pose Estimation API",
        description=(
            "Upload a video of a person performing an action and receive: "
            "(1) structured pose keypoint data, (2) a skeleton-overlay video, "
            "and (3) a standalone avatar video reproducing the movement."
        ),
        version="1.0.0",
    )
    app.include_router(router)
    return app