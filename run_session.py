"""
Classroom Attendance Session Runner.

Workflow:
1. Scans classroom over a 10-20 second temporal sampling window (or video / group photo).
2. Aggregates detections: marks a student PRESENT only if recognized in >= N frames.
3. Explicitly records all enrolled students not meeting the threshold as ABSENT.
4. Atomically commits session and attendance records to SQLite.
5. Saves an annotated proof image with bounding boxes, names, and confidence scores.
6. Auto-generates a dated CSV report in data/exports/attendance_YYYY-MM-DD_<course>.csv.
7. Supports tiling and upscaling for distant / small faces in large lecture halls.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

import db
from recognizer import Recognizer
from config import (
    DB_PATH,
    EXPORTS_DIR,
    SESSION_SAMPLE_DURATION_SEC,
    SESSION_FRAME_INTERVAL,
    SESSION_MIN_FRAMES_SEEN,
    COSINE_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_METRIC,
)


def detect_with_tiles_or_upscale(
    image: np.ndarray,
    recognizer: Recognizer,
    upscale_factor: float = 1.0,
    tile_size: Optional[int] = None,
    overlap_ratio: float = 0.20,
) -> np.ndarray:
    """
    Enhanced detection supporting upscaling and grid tiling for small distant faces.

    Args:
        image: Original BGR frame.
        recognizer: Recognizer instance.
        upscale_factor: Scaling factor (e.g. 1.5 or 2.0).
        tile_size: If specified, divides image into overlapping tiles of this dimension.
        overlap_ratio: Overlap percentage between adjacent tiles (default 20%).

    Returns:
        Consolidated numpy array of shape (N, 15) after Non-Maximum Suppression.
    """
    if image is None or image.size == 0:
        return np.empty((0, 15), dtype=np.float32)

    # 1. Direct upscaling mode
    if upscale_factor > 1.0 and not tile_size:
        h, w = image.shape[:2]
        new_w, new_h = int(w * upscale_factor), int(h * upscale_factor)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        faces = recognizer.detect(upscaled)
        if len(faces) == 0:
            return faces

        # Rescale coordinates back to original image scale
        scale = 1.0 / upscale_factor
        faces_rescaled = faces.copy()
        # Box [x, y, w, h]
        faces_rescaled[:, 0:4] *= scale
        # 5 Landmarks [x, y] pairs
        faces_rescaled[:, 4:14] *= scale
        return faces_rescaled

    # 2. Tiling mode for high-resolution group photographs
    if tile_size and tile_size > 0:
        h, w = image.shape[:2]
        step = int(tile_size * (1.0 - overlap_ratio))
        all_candidate_faces = []

        for y0 in range(0, h, step):
            y1 = min(y0 + tile_size, h)
            for x0 in range(0, w, step):
                x1 = min(x0 + tile_size, w)
                tile = image[y0:y1, x0:x1]
                tile_faces = recognizer.detect(tile)
                if len(tile_faces) > 0:
                    # Offset tile face coordinates to original frame space
                    for face in tile_faces:
                        face_global = face.copy()
                        face_global[0] += x0  # x
                        face_global[1] += y0  # y
                        for l_idx in range(5):
                            face_global[4 + 2 * l_idx] += x0
                            face_global[5 + 2 * l_idx] += y0
                        all_candidate_faces.append(face_global)

        if not all_candidate_faces:
            # Fallback to full frame detection
            return recognizer.detect(image)

        candidates = np.array(all_candidate_faces, dtype=np.float32)
        # Apply OpenCV NMS to merge detections overlapping across tiles
        boxes = [[int(c[0]), int(c[1]), int(c[2]), int(c[3])] for c in candidates]
        scores = [float(c[14]) for c in candidates]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=recognizer.score_threshold,
            nms_threshold=0.35,
        )
        if len(indices) == 0:
            return np.empty((0, 15), dtype=np.float32)

        indices = np.array(indices).flatten()
        return candidates[indices]

    # Standard detection
    return recognizer.detect(image)


def run_session_on_stream(
    cap: cv2.VideoCapture,
    course_name: str,
    recognizer: Recognizer,
    duration_sec: int = SESSION_SAMPLE_DURATION_SEC,
    min_frames_seen: int = SESSION_MIN_FRAMES_SEEN,
    frame_interval: int = SESSION_FRAME_INTERVAL,
    display: bool = True,
    upscale_factor: float = 1.0,
    tile_size: Optional[int] = None,
) -> Tuple[int, Path, Path]:
    """
    Run attendance scan across a video capture stream (webcam or video file).

    Returns:
        Tuple of (session_id, proof_image_path, csv_path).
    """
    enrolled_students = recognizer.load_enrolled_students()
    if not enrolled_students:
        print("Error: No students enrolled in database. Run 'python enroll.py' first.")
        sys.exit(1)

    print("=" * 70)
    print(f"Starting Attendance Session for Course: '{course_name}'")
    print(f"Enrolled Students: {len(enrolled_students)} | Scan Duration: {duration_sec}s")
    print(f"Minimum Frames Threshold: {min_frames_seen} frames")
    print("=" * 70)

    # Tracking structures: per-student frequency and max confidence
    student_stats = {
        s["id"]: {
            "roll_no": s["roll_no"],
            "name": s["name"],
            "frames_seen": 0,
            "max_confidence": 0.0,
        }
        for s in enrolled_students
    }

    best_proof_frame = None
    max_faces_in_proof = -1
    proof_recognition_results = []

    frame_idx = 0
    start_time = time.time()
    t_last_fps = time.time()
    fps_count = 0
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if frame_idx % frame_interval != 0:
                continue

            elapsed = time.time() - start_time
            if duration_sec > 0 and elapsed >= duration_sec:
                print(f"\nCompleted sampling window of {duration_sec} seconds.")
                break

            # 1. Detection (with optional upscaling/tiling)
            faces = detect_with_tiles_or_upscale(
                frame,
                recognizer,
                upscale_factor=upscale_factor,
                tile_size=tile_size,
            )

            # 2. Embedding & Matching per detected face
            current_frame_results = []
            for face in faces:
                emb = recognizer.embed(frame, face)
                matched_student, confidence = recognizer.match(
                    emb,
                    enrolled_students=enrolled_students,
                )
                current_frame_results.append((face, matched_student, confidence))

                if matched_student is not None:
                    sid = matched_student["id"]
                    student_stats[sid]["frames_seen"] += 1
                    if confidence > student_stats[sid]["max_confidence"]:
                        student_stats[sid]["max_confidence"] = float(confidence)

            # Keep the frame that captured the largest number of faces as proof
            if len(faces) > max_faces_in_proof or best_proof_frame is None:
                max_faces_in_proof = len(faces)
                best_proof_frame = frame.copy()
                proof_recognition_results = current_frame_results

            # Visual overlay
            annotated = recognizer.draw_recognition_results(frame, current_frame_results)

            # Overlay timer and active head count
            time_left = max(0, int(duration_sec - elapsed)) if duration_sec > 0 else 0
            overlay_str = (
                f"Course: {course_name} | Time Left: {time_left}s | "
                f"Faces Detected: {len(faces)}"
            )
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                overlay_str,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if display:
                cv2.imshow("Attendance Session Scanner (Press 'q' to end early)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("\nSession scan ended early by user.")
                    break

    finally:
        if display:
            cv2.destroyAllWindows()

    # 3. Create Session in SQLite
    session_date = datetime.now().strftime("%Y-%m-%d")
    session_time = datetime.now().strftime("%H:%M:%S")
    session_id = db.create_session(
        course=course_name,
        date=session_date,
        start_time=session_time,
    )

    # 4. Record Attendance for EVERY student (PRESENT or ABSENT)
    present_count = 0
    attendance_rows = []

    for sid, stat in student_stats.items():
        frames_seen = stat["frames_seen"]
        conf = stat["max_confidence"]

        if frames_seen >= min_frames_seen:
            status = "PRESENT"
            present_count += 1
        else:
            status = "ABSENT"
            conf = 0.0

        db.record_attendance(
            session_id=session_id,
            student_id=sid,
            status=status,
            confidence=conf,
            frames_seen=frames_seen,
        )

        attendance_rows.append(
            {
                "Roll No": stat["roll_no"],
                "Name": stat["name"],
                "Course": course_name,
                "Date": session_date,
                "Time": session_time,
                "Status": status,
                "Confidence": round(conf, 4),
                "Frames Seen": frames_seen,
            }
        )

    # 5. Save Annotated Proof Image
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_course = "".join(c if c.isalnum() else "_" for c in course_name)
    proof_filename = f"session_{session_id}_{clean_course}_proof.jpg"
    proof_path = EXPORTS_DIR / proof_filename

    if best_proof_frame is not None:
        annotated_proof = recognizer.draw_recognition_results(
            best_proof_frame, proof_recognition_results
        )
        # Add summary header to proof image
        cv2.putText(
            annotated_proof,
            f"Session #{session_id} - {course_name} ({session_date}) | Present: {present_count}/{len(enrolled_students)}",
            (10, annotated_proof.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(proof_path), annotated_proof)
    else:
        # Fallback placeholder if no frame grabbed
        placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.imwrite(str(proof_path), placeholder)

    # 6. Auto-generate CSV File
    csv_filename = f"attendance_{session_date}_{clean_course}.csv"
    csv_path = EXPORTS_DIR / csv_filename
    df = pd.DataFrame(attendance_rows)
    df.sort_values(by="Roll No", inplace=True)
    df.to_csv(csv_path, index=False)

    # 7. Print Console Summary Table
    total_enrolled = len(enrolled_students)
    absent_count = total_enrolled - present_count
    attendance_pct = (present_count / total_enrolled * 100) if total_enrolled > 0 else 0

    print("\n" + "=" * 70)
    print(f"ATTENDANCE SESSION SUMMARY (Session ID: #{session_id})")
    print("=" * 70)
    print(f"Course         : {course_name}")
    print(f"Date & Time    : {session_date} {session_time}")
    print(f"Total Enrolled : {total_enrolled}")
    print(f"Present        : {present_count} ({attendance_pct:.1f}%)")
    print(f"Absent         : {absent_count} ({100 - attendance_pct:.1f}%)")
    print(f"Proof Image    : {proof_path}")
    print(f"CSV Export     : {csv_path}")
    print("-" * 70)
    print(f"{'Roll No':<10} | {'Name':<22} | {'Status':<8} | {'Frames':<7} | {'Confidence'}")
    print("-" * 70)
    for row in attendance_rows:
        status_marker = "[P]" if row["Status"] == "PRESENT" else "[A]"
        print(
            f"{row['Roll No']:<10} | {row['Name']:<22} | {status_marker} {row['Status']:<4} | "
            f"{row['Frames Seen']:<7} | {row['Confidence']:.4f}"
        )
    print("=" * 70)

    return session_id, proof_path, csv_path


def run_session_on_image(
    image_path: Path,
    course_name: str,
    recognizer: Recognizer,
    min_frames_seen: int = 1,
    display: bool = True,
    upscale_factor: float = 1.0,
    tile_size: Optional[int] = None,
) -> Tuple[int, Path, Path]:
    """Run an attendance session on a single classroom snapshot image."""
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)

    enrolled_students = recognizer.load_enrolled_students()
    if not enrolled_students:
        print("Error: No students enrolled in database.")
        sys.exit(1)

    faces = detect_with_tiles_or_upscale(
        frame,
        recognizer,
        upscale_factor=upscale_factor,
        tile_size=tile_size,
    )

    student_stats = {
        s["id"]: {
            "roll_no": s["roll_no"],
            "name": s["name"],
            "frames_seen": 0,
            "max_confidence": 0.0,
        }
        for s in enrolled_students
    }

    recognition_results = []
    for face in faces:
        emb = recognizer.embed(frame, face)
        matched_student, conf = recognizer.match(emb, enrolled_students=enrolled_students)
        recognition_results.append((face, matched_student, conf))
        if matched_student is not None:
            sid = matched_student["id"]
            student_stats[sid]["frames_seen"] += 1
            if conf > student_stats[sid]["max_confidence"]:
                student_stats[sid]["max_confidence"] = float(conf)

    # Create session record
    session_date = datetime.now().strftime("%Y-%m-%d")
    session_time = datetime.now().strftime("%H:%M:%S")
    session_id = db.create_session(
        course=course_name,
        date=session_date,
        start_time=session_time,
    )

    present_count = 0
    attendance_rows = []

    for sid, stat in student_stats.items():
        frames_seen = stat["frames_seen"]
        conf = stat["max_confidence"]
        if frames_seen >= min_frames_seen:
            status = "PRESENT"
            present_count += 1
        else:
            status = "ABSENT"
            conf = 0.0

        db.record_attendance(
            session_id=session_id,
            student_id=sid,
            status=status,
            confidence=conf,
            frames_seen=frames_seen,
        )

        attendance_rows.append(
            {
                "Roll No": stat["roll_no"],
                "Name": stat["name"],
                "Course": course_name,
                "Date": session_date,
                "Time": session_time,
                "Status": status,
                "Confidence": round(conf, 4),
                "Frames Seen": frames_seen,
            }
        )

    # Save proof image
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_course = "".join(c if c.isalnum() else "_" for c in course_name)
    proof_path = EXPORTS_DIR / f"session_{session_id}_{clean_course}_proof.jpg"
    annotated_proof = recognizer.draw_recognition_results(frame, recognition_results)
    cv2.putText(
        annotated_proof,
        f"Session #{session_id} - {course_name} ({session_date}) | Present: {present_count}/{len(enrolled_students)}",
        (10, annotated_proof.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(proof_path), annotated_proof)

    # Save CSV
    csv_path = EXPORTS_DIR / f"attendance_{session_date}_{clean_course}.csv"
    df = pd.DataFrame(attendance_rows)
    df.sort_values(by="Roll No", inplace=True)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print(f"ATTENDANCE SESSION SUMMARY (Session ID: #{session_id})")
    print("=" * 70)
    print(f"Course         : {course_name}")
    print(f"Date & Time    : {session_date} {session_time}")
    print(f"Total Enrolled : {len(enrolled_students)}")
    print(f"Present        : {present_count}")
    print(f"Absent         : {len(enrolled_students) - present_count}")
    print(f"Proof Image    : {proof_path}")
    print(f"CSV Export     : {csv_path}")
    print("=" * 70)

    if display:
        try:
            cv2.imshow(f"Classroom Session Proof - Present: {present_count}", annotated_proof)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return session_id, proof_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Classroom Face Recognition Attendance Session"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Scan classroom via live webcam (default device 0)",
    )
    group.add_argument("--video", type=str, help="Scan classroom from video file")
    group.add_argument("--image", type=str, help="Scan classroom from static image")

    parser.add_argument(
        "--course",
        type=str,
        default="CS-101",
        help="Course code / name (default: CS-101)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=SESSION_SAMPLE_DURATION_SEC,
        help=f"Sampling duration in seconds (default: {SESSION_SAMPLE_DURATION_SEC}s)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=SESSION_MIN_FRAMES_SEEN,
        help=f"Minimum frames seen to mark PRESENT (default: {SESSION_MIN_FRAMES_SEEN})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=COSINE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {COSINE_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Upscale factor for small faces (e.g. 1.5 or 2.0)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Grid tile size for high-res group photos (e.g. 640)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display GUI window",
    )

    args = parser.parse_args()

    recognizer = Recognizer(similarity_threshold=args.threshold)

    if args.image:
        run_session_on_image(
            Path(args.image),
            course_name=args.course,
            recognizer=recognizer,
            min_frames_seen=1,
            display=not args.no_display,
            upscale_factor=args.upscale,
            tile_size=args.tile_size,
        )
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {args.video}")
            sys.exit(1)
        run_session_on_stream(
            cap,
            course_name=args.course,
            recognizer=recognizer,
            duration_sec=args.duration,
            min_frames_seen=args.min_frames,
            display=not args.no_display,
            upscale_factor=args.upscale,
            tile_size=args.tile_size,
        )
        cap.release()
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"Error: Unable to open camera {args.webcam}")
            sys.exit(1)
        run_session_on_stream(
            cap,
            course_name=args.course,
            recognizer=recognizer,
            duration_sec=args.duration,
            min_frames_seen=args.min_frames,
            display=not args.no_display,
            upscale_factor=args.upscale,
            tile_size=args.tile_size,
        )
        cap.release()


if __name__ == "__main__":
    main()
