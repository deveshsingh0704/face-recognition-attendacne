"""
Configuration module for the OpenCV Face Recognition Attendance System.

Contains central configurations, paths, thresholds, and operational parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Subdirectories for data storage
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "attendance.db"

# Model File Paths
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"

YUNET_MODEL_PATH = MODELS_DIR / YUNET_MODEL_FILENAME
SFACE_MODEL_PATH = MODELS_DIR / SFACE_MODEL_FILENAME

# Model Download Sources (Official OpenCV Zoo repositories)
MODEL_URLS = {
    YUNET_MODEL_FILENAME: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    SFACE_MODEL_FILENAME: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

# YuNet Detection Parameters
# - score_threshold: Minimum confidence score to accept a detected face (default ~0.9)
# - nms_threshold: Non-Maximum Suppression threshold to suppress overlapping boxes (default ~0.3)
# - top_k: Maximum candidate detections evaluated before NMS
YUNET_SCORE_THRESHOLD: float = 0.90
YUNET_NMS_THRESHOLD: float = 0.30
YUNET_TOP_K: int = 5000

# SFace Recognition Parameters
# Cosine Similarity: Range [-1, 1], where 1 means identical.
# SFace default threshold recommended by OpenCV authors is 0.363.
# Higher score => greater confidence / more alike.
COSINE_SIMILARITY_THRESHOLD: float = 0.363

# L2 Distance: Range [0, inf), where 0 means identical.
# SFace default threshold recommended by OpenCV authors is 1.128.
# Lower distance => greater confidence / more alike.
L2_DISTANCE_THRESHOLD: float = 1.128

# Metric selector: 'cosine' or 'l2'
DEFAULT_SIMILARITY_METRIC: str = "cosine"

# Session & Sampling Parameters
SESSION_SAMPLE_DURATION_SEC: int = 15  # Total duration to scan classroom (10-20 seconds)
SESSION_FRAME_INTERVAL: int = 3       # Process every Nth frame to reduce load
SESSION_MIN_FRAMES_SEEN: int = 3      # Minimum frames seen to mark student present

# Visualization parameters
BOX_COLOR = (0, 255, 0)        # Green for bounding box
LANDMARK_COLORS = [
    (255, 0, 0),    # Right eye - Blue
    (0, 0, 255),    # Left eye - Red
    (0, 255, 255),  # Nose tip - Yellow
    (255, 0, 255),  # Right mouth corner - Magenta
    (0, 255, 0),    # Left mouth corner - Green
]
TEXT_COLOR = (255, 255, 255)   # White
UNKNOWN_COLOR = (0, 0, 255)    # Red for unrecognized face
