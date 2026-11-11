"""
Unit tests for Face Recognition Matching Engine (Phase 3).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

import db
from recognizer import Recognizer


@pytest.fixture
def recognizer():
    return Recognizer()


@pytest.fixture
def sample_face_embedding(recognizer):
    sample_path = Path("data/test_images/sample_face.jpg")
    img = cv2.imread(str(sample_path))
    faces = recognizer.detect(img)
    assert len(faces) > 0
    return recognizer.embed(img, faces[0])


def test_empty_roster_matching(recognizer, sample_face_embedding):
    student, score = recognizer.match(sample_face_embedding, enrolled_students=[])
    assert student is None
    assert score == -1.0


def test_known_student_matching(recognizer, sample_face_embedding):
    mock_student = {
        "id": 99,
        "roll_no": "TEST01",
        "name": "Test Student",
        "embeddings": np.vstack([sample_face_embedding, sample_face_embedding * 0.98]),
    }
    student, score = recognizer.match(sample_face_embedding, enrolled_students=[mock_student])
    assert student is not None
    assert student["roll_no"] == "TEST01"
    assert pytest.approx(score, abs=1e-3) == 1.0


def test_unknown_student_rejected(recognizer, sample_face_embedding):
    # Create mock students with completely orthogonal/different random embeddings
    rng = np.random.RandomState(42)
    mock_students = [
        {
            "id": i,
            "roll_no": f"DIFF{i}",
            "name": f"Other {i}",
            "embeddings": rng.randn(4, 128).astype(np.float32),
        }
        for i in range(5)
    ]
    student, score = recognizer.match(
        sample_face_embedding,
        enrolled_students=mock_students,
        threshold=0.363,
        metric="cosine",
    )
    # Cosine score between random vector and face embedding will be low (~ -0.1 to 0.2)
    assert student is None
    assert score < 0.363


def test_threshold_direction_and_overrides(recognizer, sample_face_embedding):
    # Create slightly perturbed embedding (simulating pose variation)
    noise = np.random.RandomState(123).randn(128).astype(np.float32) * 0.5
    perturbed = sample_face_embedding + noise
    sim = recognizer.match_features(sample_face_embedding, perturbed, metric="cosine")

    mock_student = {
        "id": 1,
        "roll_no": "VARY01",
        "name": "Variation Student",
        "embeddings": perturbed.reshape(1, 128),
    }

    # Strict threshold higher than sim -> should reject
    student_strict, _ = recognizer.match(
        sample_face_embedding,
        enrolled_students=[mock_student],
        threshold=sim + 0.05,
    )
    assert student_strict is None

    # Lenient threshold lower than sim -> should accept
    student_lenient, _ = recognizer.match(
        sample_face_embedding,
        enrolled_students=[mock_student],
        threshold=sim - 0.05,
    )
    assert student_lenient is not None
    assert student_lenient["roll_no"] == "VARY01"


def test_l2_metric_matching(recognizer, sample_face_embedding):
    mock_student = {
        "id": 2,
        "roll_no": "L2_TEST",
        "name": "L2 Student",
        "embeddings": sample_face_embedding.reshape(1, 128),
    }
    # For identical embedding, L2 distance is 0.0, which is <= 1.128
    student, dist = recognizer.match(
        sample_face_embedding,
        enrolled_students=[mock_student],
        metric="l2",
    )
    assert student is not None
    assert student["roll_no"] == "L2_TEST"
    assert pytest.approx(dist, abs=1e-3) == 0.0


def test_match_all_in_frame_pipeline(recognizer):
    sample_path = Path("data/test_images/sample_face.jpg")
    img = cv2.imread(str(sample_path))

    results = recognizer.match_all_in_frame(img)
    assert len(results) >= 1
    face_row, matched_student, score = results[0]
    assert len(face_row) == 15
    assert isinstance(score, float)
    # Visual rendering should return valid image without throwing
    annotated = recognizer.draw_recognition_results(img, results)
    assert annotated.shape == img.shape
