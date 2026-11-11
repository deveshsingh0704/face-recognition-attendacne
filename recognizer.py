"""
Core Face Recognition Module using OpenCV YuNet and SFace.

Encapsulates:
1. Face Detection: cv2.FaceDetectorYN (YuNet ONNX model)
   - Dynamic input resizing per frame
   - Outputs: bounding box [x, y, w, h], 5 facial landmarks, detection score
2. Face Alignment & Embedding: cv2.FaceRecognizerSF (SFace ONNX model)
   - Align face using 5 landmarks to standard 112x112 geometry
   - Extracts 128-dimensional L2-normalized float32 embedding
3. Matching:
   - Cosine Similarity (default, HIGHER = MORE ALIKE, threshold ~0.363)
   - L2 Distance (LOWER = MORE ALIKE, threshold ~1.128)
   - Multi-embedding matching strategy: Max-similarity across poses (or centroid mean)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    YUNET_SCORE_THRESHOLD,
    YUNET_NMS_THRESHOLD,
    YUNET_TOP_K,
    COSINE_SIMILARITY_THRESHOLD,
    L2_DISTANCE_THRESHOLD,
    DEFAULT_SIMILARITY_METRIC,
    BOX_COLOR,
    LANDMARK_COLORS,
    TEXT_COLOR,
    UNKNOWN_COLOR,
)
import db


class Recognizer:
    """Encapsulates OpenCV FaceDetectorYN (YuNet) and FaceRecognizerSF (SFace)."""

    def __init__(
        self,
        yunet_path: Union[str, Path] = YUNET_MODEL_PATH,
        sface_path: Union[str, Path] = SFACE_MODEL_PATH,
        score_threshold: float = YUNET_SCORE_THRESHOLD,
        nms_threshold: float = YUNET_NMS_THRESHOLD,
        top_k: int = YUNET_TOP_K,
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
        metric: str = DEFAULT_SIMILARITY_METRIC,
        match_strategy: str = "max",
    ):
        self.yunet_path = Path(yunet_path)
        self.sface_path = Path(sface_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.metric = metric.lower()
        self.match_strategy = match_strategy.lower()  # "max" or "mean"

        # Cache of enrolled students loaded from database
        self._enrolled_students_cache: Optional[List[Dict[str, Any]]] = None

        self._validate_model_files()

        # Initialize detector with default input size; dynamically resized per frame
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )
        if self.detector is None:
            raise RuntimeError(f"Failed to create FaceDetectorYN from {self.yunet_path}")

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.sface_path),
            config="",
        )
        if self.recognizer is None:
            raise RuntimeError(f"Failed to create FaceRecognizerSF from {self.sface_path}")

    def _validate_model_files(self) -> None:
        """Ensure both ONNX model files exist on disk before initialization."""
        if not self.yunet_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at '{self.yunet_path}'. "
                "Run 'python download_models.py' first."
            )
        if not self.sface_path.exists():
            raise FileNotFoundError(
                f"SFace model not found at '{self.sface_path}'. "
                "Run 'python download_models.py' first."
            )

    # -------------------------------------------------------------------------
    # 1. Detection
    # -------------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Detect faces in an image using YuNet.

        Args:
            image: BGR image as numpy array (H, W, C).
            score_threshold: Optional override for the detection confidence threshold.

        Returns:
            np.ndarray of shape (N, 15) or empty array (0, 15).
            Row format: [x, y, w, h, 5 landmark pairs, score].
        """
        if image is None or image.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        height, width = image.shape[:2]
        # YuNet requirement: input size must match the incoming frame dimensions (w, h)
        self.detector.setInputSize((int(width), int(height)))

        if score_threshold is not None:
            self.detector.setScoreThreshold(float(score_threshold))
        else:
            self.detector.setScoreThreshold(self.score_threshold)

        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)

        return np.asarray(faces, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 2. Embedding
    # -------------------------------------------------------------------------
    def embed(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """
        Align and compute a 128-dimensional embedding from a detected face using SFace.

        Args:
            image: Original BGR image (H, W, C).
            face_row: Single face detection row (15 elements).

        Returns:
            128-dimensional float32 feature vector as 1D array of shape (128,).
        """
        if face_row is None or len(face_row) < 15:
            raise ValueError("face_row must contain at least 15 elements.")

        # 1. Align and crop face to standard 112x112 using 5 landmarks
        face_row_formatted = np.asarray(face_row, dtype=np.float32)
        aligned_face = self.recognizer.alignCrop(image, face_row_formatted)

        # 2. Extract deep feature embedding
        feature = self.recognizer.feature(aligned_face)

        # Flatten to 1D vector (128,)
        return feature.flatten().astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Matching
    # -------------------------------------------------------------------------
    def match_features(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
        metric: Optional[str] = None,
    ) -> float:
        """
        Compute similarity or distance between two 128-dimensional face embeddings.

        IMPORTANT METRIC CONVENTIONS:
        - Cosine: cv2.FaceRecognizerSF_FR_COSINE (range [-1, 1], HIGHER = MORE ALIKE)
        - L2:     cv2.FaceRecognizerSF_FR_NORM_L2 (range [0, inf), LOWER = MORE ALIKE)
        """
        used_metric = (metric or self.metric).lower()
        f1 = feat1.reshape(1, -1).astype(np.float32)
        f2 = feat2.reshape(1, -1).astype(np.float32)

        if used_metric == "cosine":
            score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)
        elif used_metric in ("l2", "norm_l2"):
            dist = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_NORM_L2)
            return float(dist)
        else:
            raise ValueError(f"Unknown metric '{used_metric}'. Use 'cosine' or 'l2'.")

    def is_match(
        self,
        score: float,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
    ) -> bool:
        """
        Evaluate if a match score satisfies the identity verification threshold.

        Directional logic:
        - Cosine Similarity: score >= threshold (HIGHER = MORE ALIKE)
        - L2 Distance: score <= threshold (LOWER = MORE ALIKE)
        """
        used_metric = (metric or self.metric).lower()
        thresh = threshold if threshold is not None else self.similarity_threshold

        if used_metric == "cosine":
            return score >= thresh
        elif used_metric in ("l2", "norm_l2"):
            l2_thresh = threshold if threshold is not None else L2_DISTANCE_THRESHOLD
            return score <= l2_thresh
        else:
            raise ValueError(f"Unknown metric '{used_metric}'.")

    def load_enrolled_students(
        self,
        db_path: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load enrolled student identities and their embeddings into cache.
        """
        if self._enrolled_students_cache is None or force_reload:
            if db_path is not None:
                self._enrolled_students_cache = db.get_all_students(Path(db_path))
            else:
                self._enrolled_students_cache = db.get_all_students()
        return self._enrolled_students_cache

    def match(
        self,
        query_vector: np.ndarray,
        enrolled_students: Optional[List[Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Match a single 128-d query embedding against all enrolled students.

        Args:
            query_vector: (128,) float32 vector.
            enrolled_students: Optional list of student records. If None, loads from cache/DB.
            threshold: Optional threshold override.
            metric: 'cosine' (default) or 'l2'.
            strategy: 'max' (best match across enrolled photos) or 'mean' (centroid).

        Returns:
            Tuple of:
            - matched_student: Dict with student info if match >= threshold, else None ("unknown")
            - best_score: The highest cosine similarity (or lowest L2 distance) observed.
        """
        students = (
            enrolled_students
            if enrolled_students is not None
            else self.load_enrolled_students()
        )

        used_metric = (metric or self.metric).lower()
        used_strategy = (strategy or self.match_strategy).lower()
        thresh = threshold if threshold is not None else self.similarity_threshold

        if not students:
            # Empty database
            worst_score = -1.0 if used_metric == "cosine" else float("inf")
            return None, worst_score

        best_student: Optional[Dict[str, Any]] = None
        # Initialize sentinel best score based on metric direction
        best_score = -float("inf") if used_metric == "cosine" else float("inf")

        q = query_vector.flatten().astype(np.float32)

        for student in students:
            embs = student["embeddings"]  # Shape: (K, 128)
            if embs is None or len(embs) == 0:
                continue

            if used_strategy == "mean":
                # Centroid matching: compute mean vector and evaluate
                centroid = np.mean(embs, axis=0)
                score = self.match_features(q, centroid, metric=used_metric)
            else:
                # "max" strategy: match against all individual enrolled photos
                # and take the best score
                scores = [
                    self.match_features(q, emb, metric=used_metric)
                    for emb in embs
                ]
                score = max(scores) if used_metric == "cosine" else min(scores)

            # Check if this student is the best match so far
            if used_metric == "cosine":
                if score > best_score:
                    best_score = score
                    best_student = student
            else:  # L2 distance (lower is better)
                if score < best_score:
                    best_score = score
                    best_student = student

        # Threshold decision: Is the best match confident enough?
        is_accepted = self.is_match(best_score, threshold=thresh, metric=used_metric)

        if is_accepted and best_student is not None:
            return best_student, best_score
        else:
            # Score was below threshold -> Student is Unknown
            return None, best_score

    def match_all_in_frame(
        self,
        frame: np.ndarray,
        enrolled_students: Optional[List[Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[np.ndarray, Optional[Dict[str, Any]], float]]:
        """
        Complete end-to-end pipeline on a single frame:
        1. Detect all faces
        2. Extract embeddings
        3. Match each face against enrolled database

        Returns:
            List of tuples: (face_row_15, matched_student_dict_or_None, confidence_score)
        """
        faces = self.detect(frame, score_threshold=score_threshold)
        results = []

        if len(faces) == 0:
            return results

        students = (
            enrolled_students
            if enrolled_students is not None
            else self.load_enrolled_students()
        )

        for face in faces:
            emb = self.embed(frame, face)
            matched_student, confidence = self.match(
                emb,
                enrolled_students=students,
                threshold=threshold,
            )
            results.append((face, matched_student, confidence))

        return results

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        faces: np.ndarray,
        labels: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw visual annotations: bounding boxes, 5 landmarks, and text tags."""
        annotated = image.copy()
        if faces is None or len(faces) == 0:
            return annotated

        for idx, face in enumerate(faces):
            x, y, w, h = [int(v) for v in face[0:4]]
            det_score = float(face[14])

            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, annotated.shape[1] - x))
            h = max(1, min(h, annotated.shape[0] - y))

            label_text = ""
            box_color = BOX_COLOR
            if labels is not None and idx < len(labels):
                label_name = labels[idx]
                if "unknown" in label_name.lower():
                    box_color = UNKNOWN_COLOR
                if confidences is not None and idx < len(confidences):
                    label_text = f"{label_name} ({confidences[idx]:.2f})"
                else:
                    label_text = label_name
            else:
                label_text = f"Face: {det_score:.2f}"

            # 1. Bounding Box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # 2. 5 Facial Landmarks
            for l_idx in range(5):
                lx = int(face[4 + 2 * l_idx])
                ly = int(face[5 + 2 * l_idx])
                color = LANDMARK_COLORS[l_idx % len(LANDMARK_COLORS)]
                cv2.circle(annotated, (lx, ly), 3, color, -1)

            # 3. Label Banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            banner_y1 = max(0, y - text_h - 8)
            banner_y2 = y
            cv2.rectangle(
                annotated,
                (x, banner_y1),
                (x + text_w + 6, banner_y2),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x + 3, y - 4),
                font,
                font_scale,
                TEXT_COLOR,
                thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def draw_recognition_results(
        self,
        frame: np.ndarray,
        recognition_results: List[Tuple[np.ndarray, Optional[Dict[str, Any]], float]],
    ) -> np.ndarray:
        """
        Render recognition results:
        - Green bounding box for recognized enrolled students with Name & Roll No.
        - Red bounding box for Unknown / Unenrolled visitors.
        - Facial landmarks on every face.
        """
        if not recognition_results:
            return frame.copy()

        faces = np.array([r[0] for r in recognition_results], dtype=np.float32)
        labels = []
        confidences = []

        for _, student, conf in recognition_results:
            if student is not None:
                labels.append(f"{student['roll_no']} {student['name']}")
            else:
                labels.append("Unknown")
            confidences.append(conf)

        return self.draw_detections(frame, faces, labels=labels, confidences=confidences)
