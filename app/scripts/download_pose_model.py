"""
Download the MediaPipe Pose Landmarker model bundle (a `.task` file).

MediaPipe's pip package does NOT ship the ML model itself -- it must
be downloaded separately from Google's model bucket and pointed to via
`POSE_MODEL_PATH` (see app/config.py). Run this once after installing
dependencies:

    python scripts/download_pose_model.py            # downloads the "lite" model
    python scripts/download_pose_model.py --variant full
    python scripts/download_pose_model.py --variant heavy

Variants trade accuracy for speed/size:
    lite  (~5 MB)  -- fastest, least accurate. Good default for an API.
    full  (~9 MB)  -- balanced.
    heavy (~29 MB) -- most accurate, slowest.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


def download(variant: str, output_dir: Path) -> Path:
    url = MODEL_URLS[variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"pose_landmarker_{variant}.task"

    print(f"Downloading {variant} model from:\n  {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved to: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(MODEL_URLS.keys()),
        default="lite",
        help="Which model size to download (default: lite).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the model in (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    try:
        destination = download(args.variant, args.output_dir)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error handling
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "If this environment blocks outbound access to "
            "storage.googleapis.com, download the file manually from the "
            "URL above on a machine that has access, then copy it into "
            f"{args.output_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\nSet this as your POSE_MODEL_PATH (or leave it -- it matches the "
        f"default): {destination}"
    )


if __name__ == "__main__":
    main()