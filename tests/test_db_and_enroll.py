"""
Unit tests for db.py and enroll.py functionality.
"""

import sqlite3
import pytest
import numpy as np
import cv2
from pathlib import Path

import db
from recognizer import Recognizer
from enroll import validate_single_face


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database path for isolated testing."""
    test_db_path = tmp_path / "test_attendance.db"
    db.init_db(test_db_path)
    return test_db_path


def test_serialization_roundtrip():
    # Test (5, 128) float32 embeddings
    original = np.random.randn(5, 128).astype(np.float32)
    blob = db.serialize_embeddings(original)
    recovered = db.deserialize_embeddings(blob)

    assert recovered.shape == (5, 128)
    assert recovered.dtype == np.float32
    assert np.allclose(original, recovered)


def test_student_crud(temp_db):
    # Insert student
    embs = np.random.randn(4, 128).astype(np.float32)
    s_id = db.insert_student("CS001", "Alice Test", embs, db_path=temp_db)
    assert s_id > 0

    # Retrieve student
    student = db.get_student_by_roll("CS001", db_path=temp_db)
    assert student is not None
    assert student["roll_no"] == "CS001"
    assert student["name"] == "Alice Test"
    assert student["embeddings"].shape == (4, 128)

    # Duplicate roll_no should raise sqlite3.IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_student("CS001", "Alice Duplicate", embs, db_path=temp_db)

    # Update embeddings
    new_embs = np.random.randn(7, 128).astype(np.float32)
    updated = db.update_student_embeddings("CS001", new_embs, db_path=temp_db)
    assert updated is True

    student_updated = db.get_student_by_roll("CS001", db_path=temp_db)
    assert student_updated["embeddings"].shape == (7, 128)

    # Delete student
    deleted = db.delete_student("CS001", db_path=temp_db)
    assert deleted is True
    assert db.get_student_by_roll("CS001", db_path=temp_db) is None


def test_session_and_attendance(temp_db):
    # Insert student
    embs = np.random.randn(3, 128).astype(np.float32)
    s_id = db.insert_student("CS002", "Bob Test", embs, db_path=temp_db)

    # Create session
    session_id = db.create_session("CS-301 Computer Vision", db_path=temp_db)
    assert session_id > 0

    # Record attendance
    db.record_attendance(
        session_id=session_id,
        student_id=s_id,
        status="PRESENT",
        confidence=0.88,
        frames_seen=12,
        db_path=temp_db,
    )

    records = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(records) == 1
    assert records[0]["roll_no"] == "CS002"
    assert records[0]["status"] == "PRESENT"
    assert records[0]["confidence"] == pytest.approx(0.88, abs=1e-2)
    assert records[0]["frames_seen"] == 12

    # Updating attendance for same (session_id, student_id) should not duplicate
    db.record_attendance(
        session_id=session_id,
        student_id=s_id,
        status="PRESENT",
        confidence=0.92,
        frames_seen=15,
        db_path=temp_db,
    )
    records_after = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(records_after) == 1
    assert records_after[0]["frames_seen"] == 15


def test_validate_single_face_edge_cases():
    rec = Recognizer()

    # Case 1: Empty / blank image (zero faces)
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    face_blank = validate_single_face(blank, rec, source_desc="blank")
    assert face_blank is None

    # Case 2: Image with 1 face
    sample_path = Path("data/test_images/sample_face.jpg")
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        face_single = validate_single_face(img, rec, source_desc="single")
        assert face_single is not None
        assert len(face_single) == 15
