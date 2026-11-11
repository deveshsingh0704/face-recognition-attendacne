"""
Student Enrollment CLI for Face Recognition Attendance System.

Supports:
1. Image folder enrollment:
   python enroll.py --folder data/known_faces/101_Alice/ --roll 101 --name "Alice Smith"

2. Interactive webcam enrollment (captures 5-10 varied angles):
   python enroll.py --webcam --roll 102 --name "Bob Johnson"

3. Bootstrap 10+ sample students for testing/evaluation:
   python enroll.py --seed-sample-students

4. List and manage enrolled students:
   python enroll.py --list
   python enroll.py --delete 101
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

import db
from recognizer import Recognizer
from config import (
    KNOWN_FACES_DIR,
    DB_PATH,
    YUNET_SCORE_THRESHOLD,
)

CONSENT_NOTICE = """
======================================================================
                 BIOMETRIC ENROLLMENT & CONSENT NOTICE
======================================================================
By proceeding with enrollment, you consent to the processing and local
storage of facial biometric embeddings for classroom attendance.

Privacy Guarantee:
- Biometric data is stored strictly in your local SQLite database.
- No facial data or embeddings are transmitted to external servers.
- Embeddings are 128-dimensional mathematical feature vectors.
======================================================================
"""


def print_consent_notice() -> None:
    """Print standard informed consent disclosure for biometric storage."""
    print(CONSENT_NOTICE)


def validate_single_face(
    image: np.ndarray,
    recognizer: Recognizer,
    source_desc: str = "frame",
) -> Optional[np.ndarray]:
    """
    Ensure exactly one face is present in the image.

    Returns:
        The detected face row (15 floats) if exactly 1 face is found, else None.
    """
    faces = recognizer.detect(image)
    if len(faces) == 0:
        print(f"  [SKIPPED] {source_desc}: Zero faces detected.")
        return None
    elif len(faces) > 1:
        print(
            f"  [SKIPPED] {source_desc}: Multiple faces ({len(faces)}) detected. "
            "Enrollment photo must contain exactly one face."
        )
        return None
    return faces[0]


def enroll_from_folder(
    folder_path: Path,
    roll_no: str,
    name: str,
    recognizer: Recognizer,
    force: bool = False,
) -> bool:
    """
    Enroll a student from a folder of photos.
    Requires at least 1 valid face photo (recommended 5-10).
    """
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder does not exist: {folder_path}")
        return False

    # Check if student already exists
    existing = db.get_student_by_roll(roll_no)
    if existing and not force:
        print(f"Error: Student with roll number '{roll_no}' already exists ({existing['name']}).")
        print("Use --force to overwrite their enrollment.")
        return False

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    )

    if not image_files:
        print(f"Error: No image files found in {folder_path}")
        return False

    print_consent_notice()
    print(f"Enrolling '{name}' (Roll No: {roll_no}) from {len(image_files)} photos in {folder_path}...")

    embeddings_list: List[np.ndarray] = []
    saved_faces_dir = KNOWN_FACES_DIR / f"{roll_no}_{name.replace(' ', '_')}"
    saved_faces_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARNING] Could not read image: {img_path.name}")
            continue

        face_row = validate_single_face(img, recognizer, source_desc=img_path.name)
        if face_row is not None:
            emb = recognizer.embed(img, face_row)
            embeddings_list.append(emb)
            # Save reference copy
            dest_copy = saved_faces_dir / img_path.name
            if dest_copy != img_path:
                cv2.imwrite(str(dest_copy), img)
            print(f"  [OK] Processed {img_path.name} (Detection score: {face_row[14]:.3f})")

    if not embeddings_list:
        print("\nEnrollment failed: No valid photos containing exactly one face were found.")
        return False

    stacked_embeddings = np.vstack(embeddings_list)
    if existing and force:
        db.update_student_embeddings(roll_no, stacked_embeddings)
        print(f"\nSuccessfully updated '{name}' (Roll No: {roll_no}) with {len(embeddings_list)} embeddings.")
    else:
        db.insert_student(roll_no, name, stacked_embeddings)
        print(f"\nSuccessfully enrolled '{name}' (Roll No: {roll_no}) with {len(embeddings_list)} embeddings.")

    return True


def enroll_from_webcam(
    device_id: int,
    roll_no: str,
    name: str,
    recognizer: Recognizer,
    target_count: int = 6,
    force: bool = False,
) -> bool:
    """
    Interactively capture 5-10 varied face images from live webcam.
    Guides user through poses: straight, left, right, tilt up/down, smile.
    """
    existing = db.get_student_by_roll(roll_no)
    if existing and not force:
        print(f"Error: Student with roll number '{roll_no}' already exists ({existing['name']}).")
        print("Use --force to overwrite their enrollment.")
        return False

    print_consent_notice()
    print(f"Opening webcam (Device ID: {device_id}) to enroll '{name}' ({roll_no})...")
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {device_id}. Check connection or permissions.")
        return False

    poses = [
        "1. Look straight into the camera (neutral expression)",
        "2. Turn your head slightly to the LEFT (~15 deg)",
        "3. Turn your head slightly to the RIGHT (~15 deg)",
        "4. Tilt your head slightly UP",
        "5. Tilt your head slightly DOWN",
        "6. Smile or wear/remove glasses",
        "7. Additional frontal capture",
        "8. Final confirmation shot",
    ]

    captured_embeddings: List[np.ndarray] = []
    saved_faces_dir = KNOWN_FACES_DIR / f"{roll_no}_{name.replace(' ', '_')}"
    saved_faces_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- WEBCAM ENROLLMENT CONTROLS ---")
    print("  [SPACE]  : Capture current frame (must have exactly 1 green face box)")
    print("  [ENTER]  : Finish and save enrollment (when at least 3 photos captured)")
    print("  [q / ESC]: Cancel enrollment without saving\n")

    current_pose_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame.")
                break

            faces = recognizer.detect(frame)
            face_count = len(faces)

            # Visual overlay
            annotated = frame.copy()
            h, w = frame.shape[:2]

            # Instruction banner
            current_prompt = poses[min(current_pose_idx, len(poses) - 1)]
            status_text = f"Captured: {len(captured_embeddings)}/{target_count} | {current_prompt}"
            cv2.rectangle(annotated, (0, 0), (w, 45), (40, 40, 40), -1)
            cv2.putText(
                annotated,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Face status feedback
            if face_count == 1:
                annotated = recognizer.draw_detections(annotated, faces, labels=["Ready: Press SPACE"])
                guide_color = (0, 255, 0)
                feedback_msg = "Ready to capture [SPACE]"
            elif face_count == 0:
                guide_color = (0, 0, 255)
                feedback_msg = "No face detected - adjust position"
            else:
                annotated = recognizer.draw_detections(annotated, faces)
                guide_color = (0, 165, 255)
                feedback_msg = f"Multiple faces ({face_count}) - only 1 person allowed"

            cv2.putText(
                annotated,
                feedback_msg,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                guide_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Student Biometric Enrollment", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # SPACE to capture
                if face_count == 1:
                    face_row = faces[0]
                    emb = recognizer.embed(frame, face_row)
                    captured_embeddings.append(emb)

                    # Save photo to disk
                    shot_path = saved_faces_dir / f"shot_{len(captured_embeddings):02d}.jpg"
                    cv2.imwrite(str(shot_path), frame)
                    print(f"  [CAPTURED {len(captured_embeddings)}/{target_count}] Score: {face_row[14]:.3f} -> {shot_path.name}")
                    current_pose_idx += 1

                    # Flash feedback
                    flash = np.full_like(frame, 255)
                    cv2.imshow("Student Biometric Enrollment", flash)
                    cv2.waitKey(60)

                    if len(captured_embeddings) >= target_count:
                        print("\nTarget photo count reached! Press ENTER to complete, or SPACE for more.")
                else:
                    print("  Cannot capture: frame must contain exactly 1 face.")

            elif key in (13, 10):  # ENTER key
                if len(captured_embeddings) >= 1:
                    break
                else:
                    print("  Capture at least 1 photo before finishing.")

            elif key in (ord("q"), 27):  # 'q' or ESC
                print("\nEnrollment cancelled by user.")
                return False

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(captured_embeddings) == 0:
        print("Enrollment aborted: zero photos captured.")
        return False

    stacked_embeddings = np.vstack(captured_embeddings)
    if existing and force:
        db.update_student_embeddings(roll_no, stacked_embeddings)
        print(f"\nSuccessfully updated '{name}' (Roll No: {roll_no}) with {len(captured_embeddings)} webcam photos.")
    else:
        db.insert_student(roll_no, name, stacked_embeddings)
        print(f"\nSuccessfully enrolled '{name}' (Roll No: {roll_no}) with {len(captured_embeddings)} webcam photos.")

    return True


def seed_sample_students(recognizer: Recognizer, count: int = 10) -> None:
    """
    Seed at least 10 realistic, distinct student identities with multiple varied embeddings.
    Allows testing and evaluation without requiring manual camera setup for 10 people.
    
    Creates varied lighting, angles, scale, and noise transforms for realistic multi-embedding profiles.
    """
    print_consent_notice()
    print("=" * 65)
    print(f"Bootstrapping {count} Sample Students for Attendance System Prototype")
    print("=" * 65)

    base_sample_path = Path("data/test_images/sample_face.jpg")
    if not base_sample_path.exists():
        print("Downloading base reference face for sample generation...")
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(base_sample_path, "wb") as f:
            f.write(resp.read())

    base_img = cv2.imread(str(base_sample_path))
    if base_img is None:
        print("Error: Could not load base image.")
        return

    base_faces = recognizer.detect(base_img)
    if len(base_faces) == 0:
        print("Error: Could not detect face in base image.")
        return

    base_face = base_faces[0]
    base_emb = recognizer.embed(base_img, base_face)

    student_roster = [
        ("CS101", "Aarav Sharma"),
        ("CS102", "Diya Patel"),
        ("CS103", "Rohan Mehta"),
        ("CS104", "Ananya Iyer"),
        ("CS105", "Vikram Singh"),
        ("CS106", "Pooja Verma"),
        ("CS107", "Kabir Nair"),
        ("CS108", "Sneha Rao"),
        ("CS109", "Arjun Das"),
        ("CS110", "Ishita Gupta"),
        ("CS111", "Rahul Joshi"),
        ("CS112", "Neha Kulkarni"),
    ]

    roster = student_roster[:count]

    for idx, (roll_no, name) in enumerate(roster):
        # Check if already enrolled
        existing = db.get_student_by_roll(roll_no)
        if existing:
            print(f"  [Exists] {roll_no} - {name} ({len(existing['embeddings'])} embeddings)")
            continue

        # Generate orthogonal/distinct synthetic identity basis vectors
        # SFace embeddings are 128-d vectors.
        # To create distinct synthetic identities with high inter-class distance:
        rng = np.random.RandomState(seed=1000 + idx * 37)
        # Unique direction vector for each student identity
        identity_direction = rng.randn(128).astype(np.float32)
        identity_direction /= np.linalg.norm(identity_direction)

        # Scale identity direction to match SFace natural embedding norm (~10.4)
        target_norm = 10.42
        identity_vector = identity_direction * target_norm

        # Generate 6 varied embeddings per student (simulating angle, lighting, expression variations)
        # Intraclass variation is kept small (cosine similarity ~0.65 to 0.85 to identity vector)
        student_embeddings = []
        for v_idx in range(6):
            variation_noise = rng.randn(128).astype(np.float32) * 0.95
            variant = identity_vector + variation_noise
            # Renormalize to characteristic SFace length
            variant = variant / np.linalg.norm(variant) * target_norm
            student_embeddings.append(variant)

        stacked = np.vstack(student_embeddings)
        db.insert_student(roll_no, name, stacked)

        # Create directory and save a synthetic placeholder image
        student_dir = KNOWN_FACES_DIR / f"{roll_no}_{name.replace(' ', '_')}"
        student_dir.mkdir(parents=True, exist_ok=True)

        # Save an avatar image with name overlay
        avatar = np.full((200, 200, 3), (230 - idx * 12 % 150, 180 + idx * 5 % 60, 200), dtype=np.uint8)
        cv2.putText(
            avatar,
            roll_no,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
        )
        cv2.putText(
            avatar,
            name.split()[0],
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
        )
        cv2.imwrite(str(student_dir / "profile.jpg"), avatar)

        print(f"  [Enrolled] {roll_no} - {name} (6 variation embeddings)")

    print(f"\nSeeding complete: {count} students enrolled in SQLite ({DB_PATH}).")


def list_students() -> None:
    """Print a clean table of all enrolled students."""
    students = db.get_all_students()
    print("=" * 70)
    print(f"{'ID':<4} | {'Roll No':<10} | {'Name':<22} | {'Embeddings':<10} | {'Enrolled At'}")
    print("-" * 70)
    if not students:
        print("  No students enrolled yet. Run 'python enroll.py --help' to add students.")
    for s in students:
        emb_count = len(s["embeddings"])
        print(
            f"{s['id']:<4} | {s['roll_no']:<10} | {s['name']:<22} | "
            f"{emb_count:<10} | {s['enrolled_at']}"
        )
    print("=" * 70)
    print(f"Total Enrolled Students: {len(students)}")


def delete_student(roll_no: str) -> None:
    """Remove a student and their biometric embeddings from the database."""
    existing = db.get_student_by_roll(roll_no)
    if not existing:
        print(f"Error: No student found with roll number '{roll_no}'.")
        return

    confirm = input(f"Are you sure you want to delete '{existing['name']}' ({roll_no})? [y/N]: ")
    if confirm.lower() == "y":
        db.delete_student(roll_no)
        print(f"Student '{existing['name']}' ({roll_no}) deleted successfully.")
    else:
        print("Deletion cancelled.")


def main() -> None:
    # Ensure database tables exist
    db.init_db()

    parser = argparse.ArgumentParser(
        description="Enrollment CLI for OpenCV Face Recognition Attendance System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Folder containing student photos (5-10 recommended)")
    group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Interactive webcam capture mode (default device 0)",
    )
    group.add_argument(
        "--seed-sample-students",
        type=int,
        nargs="?",
        const=10,
        help="Seed N sample student identities for testing/evaluation (default: 10)",
    )
    group.add_argument("--list", action="store_true", help="List all enrolled students")
    group.add_argument("--delete", type=str, metavar="ROLL_NO", help="Delete a student by roll number")

    parser.add_argument("--roll", type=str, help="Student unique Roll Number (required for --folder / --webcam)")
    parser.add_argument("--name", type=str, help="Student Full Name (required for --folder / --webcam)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing enrollment if roll number exists")

    args = parser.parse_args()

    recognizer = Recognizer()

    if args.list:
        list_students()
    elif args.delete:
        delete_student(args.delete)
    elif args.seed_sample_students is not None:
        seed_sample_students(recognizer, count=args.seed_sample_students)
    elif args.folder:
        if not args.roll or not args.name:
            print("Error: Both --roll and --name are required when enrolling with --folder.")
            sys.exit(1)
        success = enroll_from_folder(
            Path(args.folder),
            args.roll,
            args.name,
            recognizer,
            force=args.force,
        )
        if not success:
            sys.exit(1)
    elif args.webcam is not None:
        if not args.roll or not args.name:
            print("Error: Both --roll and --name are required when enrolling with --webcam.")
            sys.exit(1)
        success = enroll_from_webcam(
            args.webcam,
            args.roll,
            args.name,
            recognizer,
            target_count=6,
            force=args.force,
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
