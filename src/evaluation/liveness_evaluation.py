"""
src/evaluation/liveness_evaluation.py
Liveness & Anti-Spoofing Empirical Evaluation Module
Smart Bus Face Recognition and Pass Verification System

This module evaluates the performance, accuracy, and latency of the
active challenge-response and landmark motion variance anti-spoofing system.

Evaluates 10 controlled scenarios:
1. Live Person (Turn Left satisfied)
2. Live Person (Turn Right satisfied)
3. Printed Photo Attack (Static variance check)
4. Smartphone Photo Attack (Challenge timeout)
5. Replayed Video Attack (Challenge direction mismatch)
6. Rigid / Motionless Pose (Exceeds challenge timeout)
7. Natural Movement (Normal micro-movements without challenge response)
8. Low-Light / Occlusion (Missing landmarks)
9. Degenerate Bounding Box / Coordinates
10. Multiple Faces in Frame (Independent spatial tracking isolation)
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.liveness import (
    LivenessDetector,
    LivenessState,
    ChallengeType,
    LivenessResult
)


def make_landmarks(yaw_type: str = "NEUTRAL", noise: float = 0.0) -> np.ndarray:
    """
    Creates synthetic 5-point facial landmarks:
    0: left eye, 1: right eye, 2: nose, 3: mouth left, 4: mouth right.
    """
    pts = np.zeros((5, 2), dtype=np.float32)
    pts[0] = [60.0 + noise, 50.0]   # Left Eye
    pts[1] = [100.0 + noise, 50.0]  # Right Eye
    pts[3] = [65.0 + noise, 90.0]   # Mouth Left
    pts[4] = [95.0 + noise, 90.0]   # Mouth Right

    # Eye span = 100 - 60 = 40.0
    if yaw_type == "NEUTRAL":
        pts[2] = [80.0 + noise, 70.0]  # Yaw = (80-60)/40 = 0.50
    elif yaw_type == "TURN_LEFT":
        pts[2] = [68.0 + noise, 70.0]  # Yaw = (68-60)/40 = 0.20
    elif yaw_type == "TURN_RIGHT":
        pts[2] = [92.0 + noise, 70.0]  # Yaw = (92-60)/40 = 0.80
    return pts


def run_liveness_evaluation() -> Dict[str, Any]:
    """
    Executes controlled tests across all 10 attack and operational vectors,
    measuring state transitions, success rates, and evaluation latencies.
    """
    base_time = datetime(2026, 10, 3, 8, 30, 0, tzinfo=timezone.utc)
    box = (50, 50, 200, 200)
    test_results = []
    latencies = []

    # -------------------------------------------------------------
    # Test 1: Live Person (Turn Left Challenge)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0, min_frames=3, challenge_type="TURN_LEFT")
    t0 = time.perf_counter()
    r1 = detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time)
    r2 = detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=0.2))
    r3 = detector.evaluate(box, make_landmarks("TURN_LEFT"), current_timestamp=base_time + timedelta(seconds=0.5))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 3.0)
    test_results.append({
        "scenario": "Live Person (Turn Left)",
        "category": "Bona Fide Presentation",
        "expected_state": "LIVE",
        "observed_state": r3.state.name,
        "passed": r3.state == LivenessState.LIVE,
        "details": f"Transitioned from {r1.state.name} to {r3.state.name} upon head turn (3 frames)"
    })

    # -------------------------------------------------------------
    # Test 2: Live Person (Turn Right Challenge)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0, min_frames=3, challenge_type="TURN_RIGHT")
    t0 = time.perf_counter()
    r1 = detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time)
    r2 = detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=0.2))
    r3 = detector.evaluate(box, make_landmarks("TURN_RIGHT"), current_timestamp=base_time + timedelta(seconds=0.5))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 3.0)
    test_results.append({
        "scenario": "Live Person (Turn Right)",
        "category": "Bona Fide Presentation",
        "expected_state": "LIVE",
        "observed_state": r3.state.name,
        "passed": r3.state == LivenessState.LIVE,
        "details": f"Transitioned from {r1.state.name} to {r3.state.name} upon head turn (3 frames)"
    })


    # -------------------------------------------------------------
    # Test 3: Printed Color Photo Attack (Static Variance)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0)
    t0 = time.perf_counter()
    r = None
    pts_static = make_landmarks("NEUTRAL")
    # Feed 7 identical frames across 2 seconds
    for i in range(7):
        r = detector.evaluate(box, pts_static, current_timestamp=base_time + timedelta(seconds=i * 0.3))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 7.0)
    test_results.append({
        "scenario": "Printed Color Photo",
        "category": "Presentation Attack (2D Print)",
        "expected_state": "SPOOF_SUSPECTED",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.SPOOF_SUSPECTED,
        "details": "Flagged SPOOF_SUSPECTED due to zero landmark motion variance"
    })

    # -------------------------------------------------------------
    # Test 4: Smartphone Photo Attack (Challenge Timeout)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=1.5)
    t0 = time.perf_counter()
    r1 = detector.evaluate(box, make_landmarks("NEUTRAL", noise=0.1), current_timestamp=base_time)
    # Exceed timeout without rotation
    r2 = detector.evaluate(box, make_landmarks("NEUTRAL", noise=0.15), current_timestamp=base_time + timedelta(seconds=2.0))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 2.0)
    test_results.append({
        "scenario": "Smartphone Screen Photo",
        "category": "Presentation Attack (2D Screen)",
        "expected_state": "SPOOF_SUSPECTED",
        "observed_state": r2.state.name,
        "passed": r2.state == LivenessState.SPOOF_SUSPECTED,
        "details": "Timed out after 1.5s without completing head turn challenge"
    })

    # -------------------------------------------------------------
    # Test 5: Replayed Video Clip (Challenge Direction Mismatch)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0, challenge_type="TURN_LEFT")
    t0 = time.perf_counter()
    detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time)
    # Video turns right when challenge was left
    r = detector.evaluate(box, make_landmarks("TURN_RIGHT"), current_timestamp=base_time + timedelta(seconds=0.5))
    t1 = time.perf_counter()
    latencies.append(t1 - t0)
    test_results.append({
        "scenario": "Replayed Video Clip",
        "category": "Presentation Attack (Video Replay)",
        "expected_state": "INSUFFICIENT_DATA",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.INSUFFICIENT_DATA,
        "details": "Wrong turn direction does not satisfy challenge; access gated"
    })

    # -------------------------------------------------------------
    # Test 6: Rigid Pose / Freezing Motionless
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=1.5)
    t0 = time.perf_counter()
    detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time)
    r = detector.evaluate(box, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=2.0))
    t1 = time.perf_counter()
    latencies.append(t1 - t0)
    test_results.append({
        "scenario": "Rigid Pose / Freezing Motionless",
        "category": "Human Edge Case",
        "expected_state": "SPOOF_SUSPECTED",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.SPOOF_SUSPECTED,
        "details": "Deliberate freeze times out safely as spoof suspected"
    })

    # -------------------------------------------------------------
    # Test 7: Natural Movement (Breathing & Drift)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0)
    t0 = time.perf_counter()
    rng = np.random.RandomState(42)
    r = None
    for i in range(5):
        n = float(rng.uniform(-0.5, 0.5))
        r = detector.evaluate(box, make_landmarks("NEUTRAL", noise=n), current_timestamp=base_time + timedelta(seconds=i * 0.1))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 5.0)
    test_results.append({
        "scenario": "Natural Facial Movement",
        "category": "Bona Fide Presentation",
        "expected_state": "INSUFFICIENT_DATA",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.INSUFFICIENT_DATA,
        "details": "Sufficient variance prevents false photo alarm while waiting for turn"
    })

    # -------------------------------------------------------------
    # Test 8: Low-Light / Severe Occlusion (Missing Landmarks)
    # -------------------------------------------------------------
    detector = LivenessDetector()
    t0 = time.perf_counter()
    r = detector.evaluate(box, None, current_timestamp=base_time)
    t1 = time.perf_counter()
    latencies.append(t1 - t0)
    test_results.append({
        "scenario": "Low-Light / Occlusion",
        "category": "Environmental Edge Case",
        "expected_state": "INSUFFICIENT_DATA",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.INSUFFICIENT_DATA,
        "details": "Safely gates access and prompts passenger to face camera"
    })

    # -------------------------------------------------------------
    # Test 9: Degenerate Bounding Box
    # -------------------------------------------------------------
    detector = LivenessDetector()
    t0 = time.perf_counter()
    r = detector.evaluate((100, 100, 50, 50), make_landmarks("NEUTRAL"), current_timestamp=base_time)
    t1 = time.perf_counter()
    latencies.append(t1 - t0)
    test_results.append({
        "scenario": "Malformed Bounding Box",
        "category": "Fault Injection",
        "expected_state": "ERROR",
        "observed_state": r.state.name,
        "passed": r.state == LivenessState.ERROR,
        "details": "Invalid coordinates caught gracefully without uncaught exception"
    })

    # -------------------------------------------------------------
    # Test 10: Multiple Faces in Frame (Spatial Isolation)
    # -------------------------------------------------------------
    detector = LivenessDetector(challenge_timeout=3.0, min_frames=3, challenge_type="TURN_LEFT")
    box_p1 = (50, 50, 200, 200)
    box_p2 = (300, 50, 450, 200)
    t0 = time.perf_counter()
    # Frame 1: Both neutral
    detector.evaluate(box_p1, make_landmarks("NEUTRAL"), current_timestamp=base_time)
    detector.evaluate(box_p2, make_landmarks("NEUTRAL"), current_timestamp=base_time)

    # Frame 2: Both neutral
    detector.evaluate(box_p1, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=0.2))
    detector.evaluate(box_p2, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=0.2))

    # Frame 3: Person 1 turns left; Person 2 remains static neutral
    r_p1 = detector.evaluate(box_p1, make_landmarks("TURN_LEFT"), current_timestamp=base_time + timedelta(seconds=0.5))
    r_p2 = detector.evaluate(box_p2, make_landmarks("NEUTRAL"), current_timestamp=base_time + timedelta(seconds=0.5))
    t1 = time.perf_counter()
    latencies.append((t1 - t0) / 2.0)

    isolated = (r_p1.state == LivenessState.LIVE) and (r_p2.state == LivenessState.INSUFFICIENT_DATA)
    test_results.append({
        "scenario": "Multiple Faces Spatial Isolation",
        "category": "Multi-Passenger Scene",
        "expected_state": "P1=LIVE, P2=INSUFFICIENT",
        "observed_state": f"P1={r_p1.state.name}, P2={r_p2.state.name}",
        "passed": isolated,
        "details": "Liveness pass of Person 1 does not leak to Person 2"
    })


    avg_latency_ms = float(np.mean(latencies) * 1000.0)
    passed_count = sum(1 for t in test_results if t["passed"])

    return {
        "total_scenarios": len(test_results),
        "passed_scenarios": passed_count,
        "pass_rate": (passed_count / len(test_results)) * 100.0,
        "average_latency_ms": avg_latency_ms,
        "test_results": test_results
    }


def print_liveness_summary(res: Dict[str, Any]):
    print("\n" + "=" * 75)
    print("   LIVENESS & ANTI-SPOOFING EMPIRICAL EVALUATION REPORT")
    print("=" * 75)
    print(f"Total Test Scenarios Evaluated : {res['total_scenarios']}")
    print(f"Scenarios Passed               : {res['passed_scenarios']} / {res['total_scenarios']}")
    print(f"Overall Pass Rate              : {res['pass_rate']:.1f}%")
    print(f"Average Liveness Latency       : {res['average_latency_ms']:.4f} ms per face")
    print("-" * 75)
    print(f"{'#':<3} | {'Scenario':<32} | {'Expected':<12} | {'Observed':<12} | {'Status'}")
    print("-" * 75)
    for i, t in enumerate(res['test_results'], 1):
        status = "PASSED" if t["passed"] else "FAILED"
        print(f"{i:<3} | {t['scenario']:<32} | {t['expected_state']:<12} | {t['observed_state']:<12} | {status}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    results = run_liveness_evaluation()
    print_liveness_summary(results)
