"""
Unit tests for evaluate.py (Phase 5).
"""

import pytest
import numpy as np
from pathlib import Path

import db
from recognizer import Recognizer
from evaluate import (
    BenchmarkProbe,
    build_evaluation_benchmark,
    evaluate_benchmark,
    run_threshold_sweep,
    run_condition_breakdown,
    run_latency_benchmark,
)
from config import EXPORTS_DIR


@pytest.fixture
def recognizer():
    return Recognizer()


@pytest.fixture
def enrolled_students(recognizer):
    return recognizer.load_enrolled_students()


def test_build_benchmark_dataset(recognizer, enrolled_students):
    probes = build_evaluation_benchmark(recognizer, enrolled_students)
    assert len(probes) >= 50
    genuine = [p for p in probes if p.true_label != "UNKNOWN"]
    impostor = [p for p in probes if p.true_label == "UNKNOWN"]
    assert len(genuine) > 0
    assert len(impostor) > 0
    for p in probes:
        assert p.embedding.shape == (128,)
        assert p.embedding.dtype == np.float32


def test_evaluate_benchmark_metrics(recognizer, enrolled_students):
    probes = build_evaluation_benchmark(recognizer, enrolled_students)
    metrics = evaluate_benchmark(probes, enrolled_students, recognizer, threshold=0.363)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "far" in metrics
    assert "frr" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["far"] <= 1.0
    assert 0.0 <= metrics["frr"] <= 1.0


def test_threshold_sweep_output(recognizer, enrolled_students, tmp_path):
    probes = build_evaluation_benchmark(recognizer, enrolled_students)
    df, eer_thresh = run_threshold_sweep(
        probes,
        enrolled_students,
        recognizer,
        start=0.2,
        stop=0.5,
        step=0.1,
        export_dir=tmp_path,
    )
    assert len(df) == 4
    assert (tmp_path / "threshold_sweep.csv").exists()
    assert (tmp_path / "threshold_sweep.png").exists()
    assert 0.0 <= eer_thresh <= 1.0


def test_condition_breakdown(recognizer, enrolled_students, tmp_path):
    df_c = run_condition_breakdown(recognizer, enrolled_students, export_dir=tmp_path)
    assert len(df_c) >= 5
    assert (tmp_path / "condition_breakdown.csv").exists()
    assert "Condition" in df_c.columns
    assert "Detected" in df_c.columns
    assert "Accuracy" in df_c.columns


def test_latency_benchmark(recognizer, enrolled_students, tmp_path):
    lat = run_latency_benchmark(recognizer, enrolled_students, num_runs=5, export_dir=tmp_path)
    assert "yunet_detection_mean_ms" in lat
    assert "sface_embedding_mean_ms" in lat
    assert "e2e_frame_mean_ms" in lat
    assert (tmp_path / "latency_summary.csv").exists()
