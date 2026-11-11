"""
Phase 1 Demo: Face Detection and 5-Landmark Localization with OpenCV YuNet.

Usage examples:
  1. Detect on a static image:
     python demo_detection.py --image data/test_images/sample_face.jpg

  2. Run real-time detection on webcam:
     python demo_detection.py --webcam

  3. Detect on a video file:
     python demo_detection.py --video path/to/classroom.mp4 --output data/exports/demo_out.avi

  4. Headless mode (no GUI window, saves result):
     python demo_detection.py --image data/test_images/sample_face.jpg --no-display --output data/test_images/output.jpg
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

from recognizer import Recognizer
from config import YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD


def run_image_detection(
    image_path: Path,
    recognizer: Recognizer,
    output_path: Path = None,
    display: bool = True,
) -> None:
    """Run detection on a single image and display/save results."""
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Unable to load image: {image_path}")
        sys.exit(1)

    print(f"Processing image: {image_path} ({frame.shape[1]}x{frame.shape[0]})")
    t0 = time.perf_counter()
    faces = recognizer.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"Detected {len(faces)} face(s) in {latency_ms:.2f} ms")

    for idx, face in enumerate(faces):
        x, y, w, h = face[0:4]
        score = face[14]
        print(
            f"  Face #{idx + 1}: Box=(x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}), "
            f"Confidence={score:.4f}"
        )
        print(f"    Right Eye: ({face[4]:.1f}, {face[5]:.1f})")
        print(f"    Left Eye : ({face[6]:.1f}, {face[7]:.1f})")
        print(f"    Nose Tip : ({face[8]:.1f}, {face[9]:.1f})")
        print(f"    Mouth R  : ({face[10]:.1f}, {face[11]:.1f})")
        print(f"    Mouth L  : ({face[12]:.1f}, {face[13]:.1f})")

    annotated = recognizer.draw_detections(frame, faces)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated result saved to: {output_path}")

    if display:
        window_title = f"YuNet Face Detection ({len(faces)} faces found) - Press any key"
        try:
            cv2.imshow(window_title, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"GUI display not supported or unavailable: {e}. Output was saved to disk.")


def run_webcam_detection(
    device_id: int,
    recognizer: Recognizer,
    output_path: Path = None,
) -> None:
    """Run real-time webcam detection with FPS display."""
    print(f"Opening webcam (Device ID: {device_id})...")
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(
            f"Error: Could not open camera {device_id}. "
            "Please check camera connection or permissions."
        )
        sys.exit(1)

    print("Camera opened successfully. Press 'q' or ESC in the window to quit.")

    writer = None
    fps_counter = 0
    fps_display = 0.0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame.")
                break

            # Run detection
            faces = recognizer.detect(frame)
            annotated = recognizer.draw_detections(frame, faces)

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_start = time.time()

            # Render overlay information
            stats_text = f"FPS: {fps_display:.1f} | Faces: {len(faces)}"
            cv2.putText(
                annotated,
                stats_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            # Optional video recording
            if output_path and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("OpenCV YuNet Detection Demo (Press 'q' to exit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Webcam session closed.")


def run_video_detection(
    video_path: Path,
    recognizer: Recognizer,
    output_path: Path = None,
    display: bool = True,
) -> None:
    """Run detection on a prerecorded video file."""
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            faces = recognizer.detect(frame)
            annotated = recognizer.draw_detections(frame, faces)

            cv2.putText(
                annotated,
                f"Frame: {frame_idx}/{total_frames} | Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Video Detection (Press 'q' to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_idx} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: YuNet Face Detection Demo (Boxes & 5 Facial Landmarks)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image file")
    group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Run live webcam detection (default device 0)",
    )
    group.add_argument("--video", type=str, help="Path to input video file")

    parser.add_argument(
        "--threshold",
        type=float,
        default=YUNET_SCORE_THRESHOLD,
        help=f"Detection confidence score threshold (default: {YUNET_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated output image/video",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI window (useful for headless / script runs)",
    )

    args = parser.parse_args()

    # Instantiate recognizer with configured parameters
    recognizer = Recognizer(score_threshold=args.threshold)

    output_path = Path(args.output) if args.output else None

    if args.image:
        run_image_detection(
            Path(args.image),
            recognizer,
            output_path=output_path,
            display=not args.no_display,
        )
    elif args.webcam is not None:
        run_webcam_detection(
            args.webcam,
            recognizer,
            output_path=output_path,
        )
    elif args.video:
        run_video_detection(
            Path(args.video),
            recognizer,
            output_path=output_path,
            display=not args.no_display,
        )


if __name__ == "__main__":
    main()
