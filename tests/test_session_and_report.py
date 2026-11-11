"""
Unit tests for run_session.py and report.py (Phase 4).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

import db
from recognizer import Recognizer
from run_session import (
    run_session_on_image,
    detect_with_tiles_or_upscale,
)
from report import print_session_report, print_cumulative_student_summary
from config import EXPORTS_DIR


@pytest.fixture
def recognizer():
    return Recognizer()


def test_tiling_and_upscale_detection(recognizer):
    sample_path = Path("data/test_images/sample_face.jpg")
    if not sample_path.exists():
        pytest.skip("sample_face.jpg not found")

    img = cv2.imread(str(sample_path))
    assert img is not None

    # Test standard detection
    faces_std = recognizer.detect(img)
    assert len(faces_std) == 1

    # Test 1.5x upscaling
    faces_up = detect_with_tiles_or_upscale(img, recognizer, upscale_factor=1.5)
    assert len(faces_up) >= 1

    # Test 320x320 tiling
    faces_tiled = detect_with_tiles_or_upscale(img, recognizer, tile_size=320, overlap_ratio=0.25)
    assert len(faces_tiled) >= 1


def test_session_execution_and_exports(recognizer, tmp_path):
    sample_path = Path("data/test_images/sample_face.jpg")
    if not sample_path.exists():
        pytest.skip("sample_face.jpg not found")

    # Run session on test image
    session_id, proof_path, csv_path = run_session_on_image(
        image_path=sample_path,
        course_name="CS-301 Computer Vision",
        recognizer=recognizer,
        min_frames_seen=1,
        display=False,
    )

    assert session_id > 0
    assert proof_path.exists()
    assert csv_path.exists()

    # Verify database records
    records = db.get_session_attendance(session_id)
    students = db.get_all_students()
    assert len(records) == len(students)

    # Verify CSV contents
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == len(students)
    assert set(["Roll No", "Name", "Course", "Date", "Time", "Status", "Confidence", "Frames Seen"]).issubset(df.columns)
    assert "PRESENT" in df["Status"].values or "ABSENT" in df["Status"].values


def test_reporting_functionality():
    # Verify report executes without throwing exceptions
    sessions = db.get_all_sessions()
    if len(sessions) > 0:
        latest_id = sessions[0]["id"]
        print_session_report(latest_id, export_csv=True)
        print_cumulative_student_summary(export_csv=True)
