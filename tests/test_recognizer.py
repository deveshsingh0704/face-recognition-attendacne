"""
Unit tests for Recognizer (FaceDetectorYN & FaceRecognizerSF).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from recognizer import Recognizer
from config import YUNET_MODEL_PATH, SFACE_MODEL_PATH


@pytest.fixture(scope="module")
def recognizer():
    """Instantiate a shared recognizer for tests."""
    return Recognizer()


def test_models_exist():
    assert YUNET_MODEL_PATH.exists(), "YuNet model file is missing"
    assert SFACE_MODEL_PATH.exists(), "SFace model file is missing"


def test_empty_image_detection(recognizer):
    empty_img = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = recognizer.detect(empty_img)
    assert isinstance(faces, np.ndarray)
    assert len(faces) == 0
    assert faces.shape == (0, 15)


def test_none_image_detection(recognizer):
    faces = recognizer.detect(None)
    assert isinstance(faces, np.ndarray)
    assert len(faces) == 0


def test_sample_face_detection(recognizer):
    sample_path = Path("data/test_images/sample_face.jpg")
    if not sample_path.exists():
        pytest.skip("Sample face image not present")

    img = cv2.imread(str(sample_path))
    assert img is not None

    faces = recognizer.detect(img)
    assert len(faces) >= 1
    face = faces[0]

    # Verify 15 elements: [x, y, w, h, 5 landmark pairs, score]
    assert len(face) == 15
    score = face[14]
    assert score >= 0.85

    # Verify bounding box is within image bounds
    x, y, w, h = face[0:4]
    assert x >= 0 and y >= 0
    assert w > 20 and h > 20


def test_embedding_and_matching(recognizer):
    sample_path = Path("data/test_images/sample_face.jpg")
    if not sample_path.exists():
        pytest.skip("Sample face image not present")

    img = cv2.imread(str(sample_path))
    faces = recognizer.detect(img)
    assert len(faces) > 0

    emb1 = recognizer.embed(img, faces[0])
    assert emb1.shape == (128,)
    assert emb1.dtype == np.float32

    # Self-match cosine similarity should be very close to 1.0
    sim_self = recognizer.match_features(emb1, emb1, metric="cosine")
    assert pytest.approx(sim_self, abs=1e-3) == 1.0
    assert recognizer.is_match(sim_self, metric="cosine") is True

    # Self-match L2 distance should be very close to 0.0
    dist_self = recognizer.match_features(emb1, emb1, metric="l2")
    assert pytest.approx(dist_self, abs=1e-3) == 0.0
    assert recognizer.is_match(dist_self, metric="l2") is True


def test_drawing_annotations(recognizer):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_faces = np.array(
        [[50, 50, 80, 80, 70, 70, 110, 70, 90, 90, 75, 110, 105, 110, 0.95]],
        dtype=np.float32,
    )
    annotated = recognizer.draw_detections(img, dummy_faces, labels=["Student Alice"], confidences=[0.92])
    assert annotated.shape == img.shape
    # Check that drawing actually modified pixels
    assert np.any(annotated > 0)
