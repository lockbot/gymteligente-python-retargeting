"""
CLI entry point for local video processing without running the API

Interface adapter: command-line runner.

Lets you process a video from the terminal without spinning up the
API server -- handy for quick local testing. Reuses the exact same
use-case/container wiring as the HTTP route in `api/routes.py`; only
the "how did we get file paths and how do we report progress" parts
differ, which is exactly what should differ between two interface
adapters for the same use case.

Usage:
    python -m app.interface_adapters.cli.process_video_cli path/to/video.mp4
    python -m app.interface_adapters.cli.process_video_cli path/to/video.mp4 \
        --output-dir results/ --format csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.interface_adapters.api.container import build_process_video_use_case


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run pose estimation on a video.")
    parser.add_argument("video_path", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cli_output"),
        help="Directory to write outputs to (default: ./cli_output).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Pose data format (default: json).",
    )
    args = parser.parse_args(argv)

    if not args.video_path.is_file():
        print(f"Error: video not found at {args.video_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_data_path = args.output_dir / f"pose_data.{args.format}"
    overlay_path = args.output_dir / "skeleton_overlay.mp4"
    avatar_path = args.output_dir / "avatar.mp4"

    print(f"Processing {args.video_path} ...")
    use_case = build_process_video_use_case(
        source_video_path=str(args.video_path),
        overlay_output_path=str(overlay_path),
        avatar_output_path=str(avatar_path),
        pose_data_format=args.format,
    )
    result = use_case.execute(
        pose_data_output_path=str(pose_data_path),
        overlay_video_output_path=str(overlay_path),
        avatar_video_output_path=str(avatar_path),
    )

    detected = result.pose_sequence.detected_frame_count
    total = len(result.pose_sequence.frames)
    print(f"Done. Pose detected in {detected}/{total} frames.")
    print(f"  Pose data:        {pose_data_path}")
    print(f"  Skeleton overlay: {overlay_path}")
    print(f"  Avatar video:     {avatar_path}")


if __name__ == "__main__":
    main()