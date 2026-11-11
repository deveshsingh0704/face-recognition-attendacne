# Prototype AI-Powered Classroom Attendance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B%20%2F%204.8%2B-green.svg)](https://opencv.org/)
[![Architecture](https://img.shields.io/badge/Pipeline-YuNet%20%2B%20SFace-orange.svg)](https://github.com/opencv/opencv_zoo)
[![Database](https://img.shields.io/badge/Storage-SQLite%20%2B%20CSV-lightgrey.svg)](https://sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)](tests/)

An automated, offline-first classroom attendance system powered by computer vision and deep learning, built strictly using **pure OpenCV** without external heavy frameworks (`no dlib`, `no face_recognition` library, and `no PyTorch/TensorFlow` in production).

---

## 1. System Architecture & Workflow

```
[ Classroom Camera / Video / Photo ]
                 │
                 ▼
     [ cv2.FaceDetectorYN (YuNet) ] ── (39.4 ms on CPU)
                 │
        Bounding Boxes + 5 Facial Landmarks (Eyes, Nose, Mouth)
                 │
                 ▼
    [ cv2.FaceRecognizerSF (SFace) ]
                 │
       Affine Alignment (112x112 canonical crop)
                 │
       128-dimensional L2-Normalized Biometric Embedding (45.3 ms)
                 │
                 ▼
     [ Vectorized Cosine Matcher ]
                 │
       Max-Similarity Scan across Enrolled Student Database (1.1 ms)
                 │
                 ▼
  [ Multi-Frame Temporal Voting (N >= 3) ]
                 │
       Filters blinks, turns, motion blur & transient false accepts
                 │
                 ▼
       [ Dual Persistence Layer ]
        ├── SQLite (`data/attendance.db`): Atomic relational records
        ├── CSV Export (`data/exports/`): Timestamped administrative logs
        └── Proof Snapshot (`data/exports/`): Annotated high-res verification image
```

### Why Pure OpenCV?
1. **Zero-Compiler Installation:** Standard libraries like `dlib` require CMake, MSVC C++ toolchains, and frequently fail to build on Windows. OpenCV's DNN module installs with a single `pip install opencv-python`.
2. **High CPU Efficiency:** Runs at **~10.8 FPS** real-time throughput on standard commodity laptop CPUs (~92 ms end-to-end per frame) without requiring dedicated GPUs.
3. **Reproducibility:** Self-contained ONNX runtimes maintained by the official OpenCV Zoo.

---

## 2. Directory Structure

```
face-recognition-attendance/
├── config.py                 # Central configurations, model paths, and thresholds
├── db.py                     # SQLite relational schema, CRUD, and raw float32 BLOB serialization
├── download_models.py        # Automated ONNX downloader with checksum/integrity checks
├── recognizer.py             # Core OpenCV wrapper (FaceDetectorYN + FaceRecognizerSF)
├── enroll.py                 # Student enrollment CLI (webcam, image folder, sample seeding)
├── run_session.py            # Live lecture attendance scanner with temporal voting
├── report.py                 # Attendance analytics, session logs, and low-attendance alerts
├── evaluate.py               # Empirical evaluation suite, threshold sweeps & stress tests
├── demo_detection.py         # Standalone YuNet face detection & landmark visualizer
├── demo_recognition.py       # Standalone SFace face recognition & identification demo
├── learnings.md              # Comprehensive engineering journal and interview preparation
├── requirements.txt          # Pinned runtime dependencies
├── models/                   # ONNX model storage (gitignored)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── data/                     # Local data & biometric storage (gitignored)
│   ├── attendance.db         # SQLite transactional database
│   ├── known_faces/          # Enrolled student image folders
│   ├── test_images/          # Benchmark test images
│   └── exports/              # Auto-generated CSVs, proof snapshots, and sweep plots
└── tests/                    # Pytest test suite (24 unit tests)
    ├── test_recognizer.py
    ├── test_db_and_enroll.py
    ├── test_recognition.py
    ├── test_session_and_report.py
    └── test_evaluate.py
```

---

## 3. Installation & Setup

### Step 1: Clone Repository & Create Virtual Environment
```powershell
git clone <repository-url>
cd face-recognition-attendance

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Pinned Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Download ONNX Models
Fetch the official OpenCV Zoo pretrained weights for YuNet and SFace:
```powershell
python download_models.py
```
*Validates file integrity and ensures model compatibility.*

---

## 4. Usage Guide

### 4.1 Enrolling Students

The system enforces strict single-face validation and presents an informed biometric consent disclosure:

```powershell
# Option A: Seed 10 realistic sample students for instant testing
python enroll.py --seed-sample-students

# Option B: Enroll from an image folder (5-10 varied photos recommended)
python enroll.py --folder path/to/photos --roll CS101 --name "Aarav Sharma"

# Option C: Interactive webcam enrollment (captures varied poses)
python enroll.py --webcam --roll CS111 --name "Devesh Singh"

# View enrolled student roster
python enroll.py --list
```

### 4.2 Running an Attendance Session

```powershell
# 1. Live classroom webcam scan (15-second multi-frame aggregation)
python run_session.py --webcam --course "CS-301 Computer Vision" --duration 15 --min-frames 3

# 2. Process a classroom group snapshot or image
python run_session.py --image data/known_faces/CS111_Devesh_Singh/shot_01.jpg --course "CS-301 Computer Vision" --no-display

# 3. For distant students in large lecture halls (tiling + upscaling)
python run_session.py --image classroom_4k.jpg --course "CS-301" --tile-size 640 --upscale 1.5
```

**Outputs generated per session:**
- Transactional records in `data/attendance.db` (enforcing `UNIQUE(session_id, student_id)`).
- Dated CSV log: `data/exports/attendance_YYYY-MM-DD_<Course>.csv`.
- Visual proof image: `data/exports/session_<id>_<Course>_proof.jpg`.

### 4.3 Viewing Reports & Analytics

```powershell
# View summary of the latest attendance session
python report.py --latest

# View complete historical log of all lecture sessions
python report.py --sessions

# View cumulative attendance percentages per student (highlights <75% attendance)
python report.py --student-summary
```

### 4.4 Running Empirical Evaluation

```powershell
# Run full evaluation suite (metrics, threshold sweep, condition stress test, latency)
python evaluate.py

# Run individual evaluation components
python evaluate.py --benchmark
python evaluate.py --threshold-sweep
python evaluate.py --conditions
python evaluate.py --latency
```

---

## 5. Experimental Evaluation & Benchmark Results

Backed strictly by real experiments on a 100-probe labeled benchmark (59 genuine, 41 impostors):

### Verification & Identification Metrics (Default Threshold $\tau = 0.363$)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **100.0%** | All 100 genuine & impostor probes correctly handled |
| **Precision** | **100.0%** | Zero false identifications |
| **Recall / Sensitivity** | **100.0%** | All enrolled students recognized |
| **F1-Score** | **1.0000** | Perfect harmonic mean |
| **False Accept Rate (FAR)** | **0.0%** | 0 / 41 impostors accepted |
| **False Reject Rate (FRR)** | **0.0%** | 0 / 59 genuine attempts rejected |
| **Equal Error Rate (EER)** | **~0.34** | Operating threshold closely matches OpenCV default ($\tau = 0.363$) |

### Threshold Sweep Curve

The empirical sweep across thresholds $\tau \in [0.10, 0.90]$ is plotted below:

![Threshold Sweep and Error Trade-off](data/exports/threshold_sweep.png)

### Real-World Stress Test Breakdown

| Condition | Detected? | Prediction | Confidence | Status | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Normal Frontal)** | Yes | `CS111` | 0.9925 | `[PASS]` | Frontal pose achieves near-perfect identity match. |
| **Low Light (0.35x Brightness)** | Yes | `CS111` | 0.8780 | `[PASS]` | YuNet remains sensitive; SFace confidence slightly drops. |
| **High Brightness (1.65x Gain)** | Yes | `CS111` | 0.9387 | `[PASS]` | Diffuse highlights preserve landmark geometry. |
| **Minor Head Tilt (+15° Roll)** | Yes | `CS111` | 0.8716 | `[PASS]` | 5-point affine alignment normalizes planar tilt. |
| **Moderate Head Tilt (+30° Roll)** | No | `UNKNOWN` | 0.0000 | `[FAIL]` | **Limitation:** Steep angles drop below YuNet detector confidence. |
| **Distant Small Face (~40px Face)**| Yes | `CS111` | 0.7977 | `[PASS]` | Detectable, but benefits from `--upscale` or `--tile-size`. |
| **Occlusion: Face Mask** | No | `UNKNOWN` | 0.0000 | `[FAIL]` | **Limitation:** Obscured mouth corners prevent landmark registration. |
| **Occlusion: Dark Sunglasses** | No | `UNKNOWN` | 0.0000 | `[FAIL]` | **Limitation:** Obscured eye centers prevent affine warping. |
| **Multi-Face Group Frame** | 2 faces | `CS111, UNK` | 0.9900 | `[PASS]` | Concurrently identifies known and rejects unknown. |
| **Impostor Presentation** | Yes | `UNKNOWN` | 0.3275 | `[PASS]` | Max similarity (0.3275) stays well below 0.363 threshold. |

### CPU Latency Profile (50 Iterations)

- **YuNet Face Detection:** 39.37 ms (±31.0 ms)
- **SFace 128-d Embedding:** 45.33 ms (±37.2 ms)
- **Database Vector Scan (11 Students):** 1.13 ms (±0.15 ms)
- **End-to-End Frame Latency:** 92.93 ms (**~10.8 FPS real-time CPU throughput**)

---

## 6. Biometric Ethics & Security

1. **Informed Consent Disclosure:** Every enrollment route displays a clear consent notice detailing biometric usage.
2. **Offline Local Storage:** Biometric vectors are stored strictly in `data/attendance.db` as raw float32 BLOBs and never transmitted over any network.
3. **Decoupled Images:** Raw enrollment photos may be archived or removed; the system performs all matching exclusively on 128-dimensional embedding vectors.

---

## 7. Automated Test Suite

Run the complete 24-test unit test suite covering database transactions, single-face validation, recognition thresholds, temporal aggregation, report generation, and evaluation:

```powershell
python -m pytest tests/ -v
```

```
tests/test_db_and_enroll.py::test_serialization_roundtrip PASSED         [  4%]
tests/test_db_and_enroll.py::test_student_crud PASSED                    [  8%]
tests/test_db_and_enroll.py::test_session_and_attendance PASSED          [ 12%]
tests/test_db_and_enroll.py::test_validate_single_face_edge_cases PASSED [ 16%]
tests/test_evaluate.py::test_build_benchmark_dataset PASSED              [ 20%]
tests/test_evaluate.py::test_evaluate_benchmark_metrics PASSED           [ 25%]
tests/test_evaluate.py::test_threshold_sweep_output PASSED               [ 29%]
tests/test_evaluate.py::test_condition_breakdown PASSED                  [ 33%]
tests/test_evaluate.py::test_latency_benchmark PASSED                    [ 37%]
tests/test_recognition.py::test_empty_roster_matching PASSED             [ 41%]
tests/test_recognition.py::test_known_student_matching PASSED            [ 45%]
tests/test_recognition.py::test_unknown_student_rejected PASSED          [ 50%]
tests/test_recognition.py::test_threshold_direction_and_overrides PASSED [ 54%]
tests/test_recognition.py::test_l2_metric_matching PASSED                [ 58%]
tests/test_recognition.py::test_match_all_in_frame_pipeline PASSED       [ 62%]
tests/test_recognizer.py::test_models_exist PASSED                       [ 66%]
tests/test_recognizer.py::test_empty_image_detection PASSED              [ 70%]
tests/test_recognizer.py::test_none_image_detection PASSED               [ 75%]
tests/test_recognizer.py::test_sample_face_detection PASSED              [ 79%]
tests/test_recognizer.py::test_embedding_and_matching PASSED             [ 83%]
tests/test_recognizer.py::test_drawing_annotations PASSED                [ 87%]
tests/test_session_and_report.py::test_tiling_and_upscale_detection PASSED [ 91%]
tests/test_session_and_report.py::test_session_execution_and_exports PASSED [ 95%]
tests/test_session_and_report.py::test_reporting_functionality PASSED    [100%]

============================= 24 passed in 8.21s ==============================
```
