"""
Model Downloader for OpenCV YuNet (Face Detection) and SFace (Face Recognition).

Fetches the official ONNX models from the OpenCV Zoo repository if not already present.
Validates model integrity by instantiating the OpenCV DNN face detection and
face recognition modules.
"""

import sys
import urllib.request
from pathlib import Path
import cv2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from config import (
    MODELS_DIR,
    MODEL_URLS,
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
)


class DownloadProgressBar:
    """Helper for displaying download progress using tqdm or basic console output."""

    def __init__(self, description: str):
        self.pbar = None
        self.description = description

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if HAS_TQDM:
            if self.pbar is None:
                self.pbar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=self.description,
                )
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.total = total_size
                self.pbar.refresh()
                self.pbar.close()
        else:
            if block_num % 50 == 0 or (block_num * block_size) >= total_size:
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                percent = (downloaded_mb / total_mb * 100) if total_mb > 0 else 0
                print(
                    f"\r[{self.description}] {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )


def download_file(url: str, destination: Path) -> None:
    """Download a file with user-agent headers and progress indication."""
    print(f"\nDownloading: {destination.name}")
    print(f"Source URL : {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(".tmp")

    # Use urllib with a standard browser User-Agent to avoid raw-CDN blocks
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 64 * 1024

        progress = DownloadProgressBar(destination.name)
        with open(temp_dest, "wb") as f:
            downloaded = 0
            block_num = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                block_num += 1
                progress(block_num, block_size, total_size)

    if temp_dest.exists():
        temp_dest.replace(destination)
    print(f"\nSaved successfully: {destination} ({destination.stat().st_size / 1024:.1f} KB)")


def verify_models() -> bool:
    """Verify that both ONNX models exist and can be loaded by OpenCV."""
    print("\n[Integrity Check] Verifying model loading via OpenCV DNN...")
    success = True

    # 1. Verify YuNet Detector
    if not YUNET_MODEL_PATH.exists():
        print(f"ERROR: YuNet model missing at {YUNET_MODEL_PATH}")
        return False
    try:
        detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        if detector is None:
            raise RuntimeError("cv2.FaceDetectorYN.create returned None.")
        print(f"[OK] YuNet detector loaded successfully ({YUNET_MODEL_PATH.name})")
    except Exception as e:
        print(f"[FAIL] YuNet initialization failed: {e}")
        success = False

    # 2. Verify SFace Recognizer
    if not SFACE_MODEL_PATH.exists():
        print(f"ERROR: SFace model missing at {SFACE_MODEL_PATH}")
        return False
    try:
        recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config="",
        )
        if recognizer is None:
            raise RuntimeError("cv2.FaceRecognizerSF.create returned None.")
        print(f"[OK] SFace recognizer loaded successfully ({SFACE_MODEL_PATH.name})")
    except Exception as e:
        print(f"[FAIL] SFace initialization failed: {e}")
        success = False

    return success


def main() -> None:
    """Main entrypoint to download and verify models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("OpenCV Face Detection & Recognition Model Downloader")
    print(f"Destination Directory: {MODELS_DIR}")
    print("=" * 60)

    for filename, url in MODEL_URLS.items():
        target_path = MODELS_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 10000:
            print(f"[Cached] {filename} already exists ({target_path.stat().st_size / 1024:.1f} KB)")
        else:
            download_file(url, target_path)

    if verify_models():
        print("\nAll models downloaded and verified successfully!")
        print("You are ready for Phase 1 face detection.")
    else:
        print("\nModel verification failed! Please check your network and OpenCV installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
