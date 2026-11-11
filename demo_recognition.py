"""
Phase 3 Demo: Live & Image-based Face Recognition with SFace Matching.

Matches detected faces against students enrolled in SQLite.
Annotates recognized students in GREEN and unknown individuals in RED.

Usage:
  1. Test recognition on an image:
     python demo_recognition.py --image data/test_images/sample_face.jpg --output data/test_images/rec_out.jpg

  2. Run real-time webcam recognition:
     python demo_recognition.py --webcam

  3. Custom similarity threshold:
     python demo_recognition.py --image data/test_images/sample_face.jpg --threshold 0.40
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

import db
from recognizer import Recognizer
from config import COSINE_SIMILARITY_THRESHOLD, DEFAULT_SIMILARITY_METRIC


def run_image_recognition(
    image_path: Path,
    recognizer: Recognizer,
    output_path: Path = None,
    display: bool = True,
) -> None:
    """Run face detection and recognition on a static image file."""
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Unable to load image: {image_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    results = recognizer.match_all_in_frame(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"Processed image: {image_path.name} in {latency_ms:.2f} ms")
    print(f"Detected {len(results)} face(s):")

    for idx, (face, student, confidence) in enumerate(results):
        box = [f"{v:.1f}" for v in face[0:4]]
        if student is not None:
            print(
                f"  Face #{idx + 1}: MATCH -> {student['roll_no']} ({student['name']}) "
                f"| Similarity: {confidence:.4f} (Threshold: {recognizer.similarity_threshold})"
            )
        else:
            print(
                f"  Face #{idx + 1}: UNKNOWN | Highest similarity: {confidence:.4f} "
                f"(Below threshold: {recognizer.similarity_threshold})"
            )

    annotated = recognizer.draw_recognition_results(frame, results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated result saved to: {output_path}")

    if display:
        try:
            cv2.imshow("Face Recognition Demo - Press any key", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"GUI display not available: {e}. Output was saved.")


def run_webcam_recognition(
    device_id: int,
    recognizer: Recognizer,
    output_path: Path = None,
) -> None:
    """Run real-time webcam face recognition."""
    enrolled = recognizer.load_enrolled_students()
    print(f"Loaded {len(enrolled)} enrolled students from database.")
    print(f"Opening webcam (Device ID: {device_id}). Press 'q' or ESC to exit...")

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {device_id}.")
        sys.exit(1)

    writer = None
    fps_counter = 0
    fps_display = 0.0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            results = recognizer.match_all_in_frame(frame, enrolled_students=enrolled)
            annotated = recognizer.draw_recognition_results(frame, results)

            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_start = time.time()

            # Status header
            header_text = (
                f"FPS: {fps_display:.1f} | Faces: {len(results)} | "
                f"Threshold: {recognizer.similarity_threshold:.2f}"
            )
            cv2.putText(
                annotated,
                header_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if output_path and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))
            if writer:
                writer.write(annotated)

            cv2.imshow("Real-Time Face Recognition (OpenCV YuNet + SFace)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Webcam session closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Face Recognition & Matching with SFace"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image file")
    group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Run live webcam recognition (default device 0)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=COSINE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {COSINE_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_SIMILARITY_METRIC,
        choices=["cosine", "l2"],
        help="Similarity metric: 'cosine' (higher=alike) or 'l2' (lower=alike)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Multi-embedding strategy: 'max' (best score across photos) or 'mean' (centroid)",
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
        help="Do not display GUI window (useful for headless / script runs)",
    )

    args = parser.parse_args()

    recognizer = Recognizer(
        similarity_threshold=args.threshold,
        metric=args.metric,
        match_strategy=args.strategy,
    )

    output_path = Path(args.output) if args.output else None

    if args.image:
        run_image_recognition(
            Path(args.image),
            recognizer,
            output_path=output_path,
            display=not args.no_display,
        )
    elif args.webcam is not None:
        run_webcam_recognition(
            args.webcam,
            recognizer,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
