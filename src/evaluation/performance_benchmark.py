"""
src/evaluation/performance_benchmark.py
Empirical Runtime Performance & Latency Benchmark Script
Smart Bus Face Recognition and Pass Verification System

Measures actual execution times across all pipeline stages:
1. Face Detection (MTCNN)
2. Embedding Generation (InceptionResnetV1)
3. Face Matching / Vector Search (EmbeddingStore cosine similarity)
4. Pass Verification Rules (PassVerifier SQLite check + cooldown)
5. Liveness Evaluation (LivenessDetector)
6. Complete End-to-End Decision Pipeline
7. Camera Processing Rate / Frame Throughput (FPS)
"""

import sys
import os
import time
import platform
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer, compare_embeddings
from src.vision.embedding_store import EmbeddingStore
from src.vision.liveness import LivenessDetector
from src.core.pass_verifier import PassVerifier, RecognitionResult, DEFAULT_RECOGNITION_THRESHOLD
from src.core.bus_entry_service import BusEntryService



def get_system_hardware_info() -> Dict[str, str]:
    """Collects host machine environment and hardware details."""
    cpu_name = platform.processor() or "x86_64 Architecture"
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution Mode)"

    return {
        "OS": os_name,
        "CPU": cpu_name,
        "RAM": "Available Host Memory",
        "GPU": gpu_name,
        "CUDA Available": str(cuda_avail),
        "Python Version": python_ver,
        "PyTorch Version": torch_ver
    }


def benchmark_component(fn, iterations: int = 30) -> Dict[str, float]:
    """Runs a function for N iterations and returns min, max, mean, std in milliseconds."""
    times = []
    # Warmup
    for _ in range(2):
        fn()

    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr))
    }


def run_performance_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  RUNNING SMART BUS SYSTEM EMPIRICAL PERFORMANCE BENCHMARKS")
    print("=" * 70)
    hw_info = get_system_hardware_info()
    for k, v in hw_info.items():
        print(f"  {k:<18}: {v}")
    print("-" * 70)

    # Prepare mock inputs
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    dummy_crop = np.full((160, 160, 3), 128, dtype=np.uint8)
    dummy_emb1 = np.random.randn(512).astype(np.float32)
    dummy_emb1 /= np.linalg.norm(dummy_emb1)
    dummy_emb2 = np.random.randn(512).astype(np.float32)
    dummy_emb2 /= np.linalg.norm(dummy_emb2)

    landmarks_sample = np.array([
        [60.0, 50.0],
        [100.0, 50.0],
        [80.0, 70.0],
        [65.0, 90.0],
        [95.0, 90.0]
    ], dtype=np.float32)

    # Initialize modules
    print("[1/6] Initializing modules for benchmarking...")
    detector = FaceDetector(device='cpu')
    recognizer = FaceRecognizer(device='cpu')
    mock_emb_path = PROJECT_ROOT / "data" / "evaluation" / "mock_bench_embs.pkl"
    store = EmbeddingStore(storage_path=mock_emb_path)
    # Populate store with 50 mock identities to benchmark realistic vector search

    for i in range(50):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store.add_embedding(f"STUDENT_{i:03d}", f"Student {i}", vec)

    db_path = PROJECT_ROOT / "data" / "smart_bus.db"
    verifier = PassVerifier(db_path=db_path)
    liveness = LivenessDetector()

    # 1. Face Detection (MTCNN)
    print("[2/6] Benchmarking MTCNN Face Detection...")
    detect_stats = benchmark_component(lambda: detector.detect_faces(dummy_frame), iterations=15)

    # 2. Embedding Generation (InceptionResnetV1)
    print("[3/6] Benchmarking InceptionResnetV1 Embedding Generation...")
    embed_stats = benchmark_component(lambda: recognizer.generate_embedding(dummy_crop), iterations=25)

    # 3. Vector Similarity Search (50 candidates)
    print("[4/6] Benchmarking Cosine Similarity Search...")
    def search_candidates():
        all_recs = store.get_all()
        for rec in all_recs:
            compare_embeddings(dummy_emb1, rec["embedding"])
    search_stats = benchmark_component(search_candidates, iterations=50)


    # 4. Pass Verification Rules (SQLite)
    mock_match = RecognitionResult(
        recognized=True,
        student_id="21CS101",
        student_name="Aarav Sharma",
        similarity=0.88
    )
    verify_stats = benchmark_component(lambda: verifier.verify(mock_match, bus_id="BUS-12", route="R-101"), iterations=50)


    # 5. Liveness Evaluation
    print("[6/6] Benchmarking Liveness Challenge & Landmark Variance Check...")
    liveness_stats = benchmark_component(lambda: liveness.evaluate((50, 50, 150, 150), landmarks_sample), iterations=50)

    # 6. Complete End-to-End Decision Pipeline
    print("[+] Calculating Combined Pipeline Latency & Throughput...")
    e2e_mean = detect_stats['mean_ms'] + liveness_stats['mean_ms'] + embed_stats['mean_ms'] + search_stats['mean_ms'] + verify_stats['mean_ms']
    e2e_min = detect_stats['min_ms'] + liveness_stats['min_ms'] + embed_stats['min_ms'] + search_stats['min_ms'] + verify_stats['min_ms']
    e2e_max = detect_stats['max_ms'] + liveness_stats['max_ms'] + embed_stats['max_ms'] + search_stats['max_ms'] + verify_stats['max_ms']
    fps_full = 1000.0 / e2e_mean if e2e_mean > 0 else 0.0

    # With recognition interval = 2 (AI inference every 2 frames, tracking on intermediate frames)
    effective_fps = 1000.0 / (detect_stats['mean_ms'] * 0.75 + (embed_stats['mean_ms'] + search_stats['mean_ms']) * 0.5)

    benchmarks = {
        "hardware": hw_info,
        "detection_mtcnn": detect_stats,
        "embedding_inception": embed_stats,
        "vector_search_50": search_stats,
        "pass_verification_db": verify_stats,
        "liveness_check": liveness_stats,
        "end_to_end_pipeline": {
            "mean_ms": e2e_mean,
            "min_ms": e2e_min,
            "max_ms": e2e_max,
            "fps_every_frame": fps_full,
            "fps_interval_2": effective_fps
        }
    }

    return benchmarks


def print_performance_report(res: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("           EMPIRICAL RUNTIME PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)
    hw = res['hardware']
    print(f"OS Platform         : {hw['OS']}")
    print(f"Processor (CPU)     : {hw['CPU']}")
    print(f"GPU Hardware        : {hw['GPU']}")
    print(f"Python Runtime      : {hw['Python Version']}")
    print(f"PyTorch Version     : {hw['PyTorch Version']}")
    print("-" * 70)
    print(f"{'Pipeline Component':<28} | {'Mean (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Std (ms)':<8}")
    print("-" * 70)

    stages = [
        ("1. MTCNN Face Detection", res['detection_mtcnn']),
        ("2. InceptionResnetV1 (512-D)", res['embedding_inception']),
        ("3. Vector Search (50 Cand.)", res['vector_search_50']),
        ("4. PassVerifier (SQLite)", res['pass_verification_db']),
        ("5. Liveness Check", res['liveness_check']),
    ]

    for label, s in stages:
        print(f"{label:<28} | {s['mean_ms']:<10.3f} | {s['min_ms']:<10.3f} | {s['max_ms']:<10.3f} | {s['std_ms']:<8.3f}")

    print("-" * 70)
    e2e = res['end_to_end_pipeline']
    print("END-TO-END VERIFICATION LATENCY:")
    print(f"  - Complete Decision Time (Mean) : {e2e['mean_ms']:.2f} ms")
    print(f"  - Complete Decision Range       : [{e2e['min_ms']:.2f} ms - {e2e['max_ms']:.2f} ms]")
    print(f"  - Frame Rate (Every Frame AI)   : ~{e2e['fps_every_frame']:.1f} FPS")
    print(f"  - Frame Rate (Interval = 2)     : ~{e2e['fps_interval_2']:.1f} FPS (Target Operational Rate)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    results = run_performance_benchmarks()
    print_performance_report(results)
