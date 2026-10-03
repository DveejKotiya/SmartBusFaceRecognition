"""
Unit Tests for Anti-Spoofing and Liveness Detection Module
Smart Bus Face Recognition and Pass Verification System

Covers:
1. Valid live-state input (challenge-response verified)
2. Insufficient data (missing landmarks / single frame)
3. Spoof / failed challenge (motionless or wrong turn)
4. Multiple-frame processing
5. Timeout enforcement
6. Invalid inputs (corrupt boxes / shapes)
7. Detector exception handling (safe failure modes)
8. State isolation between different people (spatial tracking)
9. Repeated attempts after reset
10. Challenge verification expiration (fresh challenge required)
11. Passive static photo detection (zero landmark variance)
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.liveness import (
    LivenessDetector,
    LivenessResult,
    LivenessState,
    ChallengeType
)


def make_landmarks(yaw_type: str = "NEUTRAL") -> np.ndarray:
    """
    Creates synthetic 5-point facial landmarks:
    0: left eye, 1: right eye, 2: nose, 3: mouth left, 4: mouth right.
    """
    pts = np.zeros((5, 2), dtype=np.float32)
    pts[0] = [60.0, 50.0]   # Left Eye
    pts[1] = [100.0, 50.0]  # Right Eye
    pts[3] = [65.0, 90.0]   # Mouth Left
    pts[4] = [95.0, 90.0]   # Mouth Right

    # Eye span = 100 - 60 = 40.0
    if yaw_type == "NEUTRAL":
        # Nose at 80.0 -> (80 - 60)/40 = 0.50 (Neutral)
        pts[2] = [80.0, 70.0]
    elif yaw_type == "TURN_LEFT":
        # Nose at 68.0 -> (68 - 60)/40 = 0.20 (<= 0.32 Turn Left)
        pts[2] = [68.0, 70.0]
    elif yaw_type == "TURN_RIGHT":
        # Nose at 92.0 -> (92 - 60)/40 = 0.80 (>= 0.68 Turn Right)
        pts[2] = [92.0, 70.0]
    return pts


class TestLivenessDetector(unittest.TestCase):
    """Test suite covering anti-spoofing verification and state management."""

    def setUp(self):
        self.detector = LivenessDetector(
            enabled=True,
            challenge_timeout=3.0,
            min_frames=3,
            challenge_type="TURN_LEFT"
        )
        self.box = (50, 50, 150, 150)
        self.now = datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc)

    # -----------------------------------------------------------------
    # 1. Valid Live-State Input
    # -----------------------------------------------------------------
    def test_valid_live_state_input(self):
        """Verifies that a passenger completing the requested challenge passes liveness."""
        # Frame 1: Neutral baseline
        r1 = self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)
        self.assertFalse(r1.is_live)
        self.assertEqual(r1.state, LivenessState.INSUFFICIENT_DATA)

        # Frame 2: Still neutral
        t2 = self.now + timedelta(seconds=0.2)
        r2 = self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), t2)
        self.assertFalse(r2.is_live)

        # Frame 3: Passenger turns head to the left
        t3 = self.now + timedelta(seconds=0.5)
        r3 = self.detector.evaluate(self.box, make_landmarks("TURN_LEFT"), t3)
        self.assertTrue(r3.is_live)
        self.assertEqual(r3.state, LivenessState.LIVE)
        self.assertGreaterEqual(r3.confidence, 0.90)
        self.assertIn("verified", r3.reason.lower())

    # -----------------------------------------------------------------
    # 2. Insufficient Data
    # -----------------------------------------------------------------
    def test_insufficient_data_missing_landmarks(self):
        """Verifies that missing landmarks return INSUFFICIENT_DATA without crashing."""
        res = self.detector.evaluate(self.box, landmarks=None, current_timestamp=self.now)
        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.INSUFFICIENT_DATA)
        self.assertIn("No facial landmarks", res.reason)

    # -----------------------------------------------------------------
    # 3. Spoof / Failed Challenge
    # -----------------------------------------------------------------
    def test_spoof_failed_challenge_wrong_direction(self):
        """Passenger turns in the opposite direction from the challenge."""
        # Challenge is TURN_LEFT, passenger turns RIGHT
        self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)
        t_timeout = self.now + timedelta(seconds=3.2)
        r_fail = self.detector.evaluate(self.box, make_landmarks("TURN_RIGHT"), t_timeout)

        self.assertFalse(r_fail.is_live)
        self.assertEqual(r_fail.state, LivenessState.SPOOF_SUSPECTED)
        self.assertIn("timed out", r_fail.reason.lower())

    # -----------------------------------------------------------------
    # 4. Multiple Frame Processing
    # -----------------------------------------------------------------
    def test_multiple_frame_processing_tracks_observations(self):
        """Verifies that sequential frames accumulate in the session observation list."""
        for i in range(5):
            t = self.now + timedelta(seconds=0.1 * i)
            # Add small random jitter
            pts = make_landmarks("NEUTRAL") + (np.random.rand(5, 2) * 0.05)
            self.detector.evaluate(self.box, pts, t)

        self.assertEqual(self.detector.get_active_session_count(), 1)
        sess = list(self.detector.sessions.values())[0]
        self.assertEqual(len(sess.observations), 5)

    # -----------------------------------------------------------------
    # 5. Challenge Timeout
    # -----------------------------------------------------------------
    def test_challenge_timeout_transition(self):
        """Verifies that exceeding challenge_timeout marks SPOOF_SUSPECTED."""
        # Initial frame
        self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)

        # Later frame after 3.5 seconds without turning head
        t_late = self.now + timedelta(seconds=3.5)
        res = self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), t_late)

        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.SPOOF_SUSPECTED)
        self.assertIn("timed out", res.reason.lower())

    # -----------------------------------------------------------------
    # 6. Invalid Inputs
    # -----------------------------------------------------------------
    def test_invalid_bounding_box(self):
        """Verifies that corrupt or degenerate bounding boxes return ERROR."""
        # box where x2 <= x1
        bad_box = (100, 50, 50, 150)
        res = self.detector.evaluate(bad_box, make_landmarks("NEUTRAL"), self.now)
        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.ERROR)

    def test_invalid_landmark_shape(self):
        """Verifies that malformed landmark arrays return ERROR safely."""
        bad_pts = np.zeros((3, 2), dtype=np.float32)  # Should be (5, 2)
        res = self.detector.evaluate(self.box, bad_pts, self.now)
        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.ERROR)

    # -----------------------------------------------------------------
    # 7. Detector Exception Handling (Safe Failure)
    # -----------------------------------------------------------------
    def test_detector_safe_failure_on_exception(self):
        """Verifies that unexpected calculations fail safely (never allow entry)."""
        # Array with NaN coordinates
        nan_pts = np.full((5, 2), np.nan, dtype=np.float32)
        res = self.detector.evaluate(self.box, nan_pts, self.now)
        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.ERROR)

    # -----------------------------------------------------------------
    # 8. State Reset Between Different People (No State Leakage)
    # -----------------------------------------------------------------
    def test_state_isolation_between_different_people(self):
        """
        Verifies that Person A passing liveness does not accidentally grant
        liveness to a different Person B in another part of the frame.
        """
        # Person A passes challenge at box (50, 50, 150, 150)
        box_a = (50, 50, 150, 150)
        self.detector.evaluate(box_a, make_landmarks("NEUTRAL"), self.now)
        self.detector.evaluate(box_a, make_landmarks("NEUTRAL"), self.now + timedelta(seconds=0.1))
        res_a = self.detector.evaluate(box_a, make_landmarks("TURN_LEFT"), self.now + timedelta(seconds=0.3))
        self.assertTrue(res_a.is_live)

        # Person B arrives at a completely separate location (400, 300, 500, 400)
        box_b = (400, 300, 500, 400)
        res_b = self.detector.evaluate(box_b, make_landmarks("NEUTRAL"), self.now + timedelta(seconds=0.4))

        # Person B MUST start a fresh challenge and must NOT be marked live!
        self.assertFalse(res_b.is_live)
        self.assertEqual(res_b.state, LivenessState.INSUFFICIENT_DATA)
        self.assertEqual(self.detector.get_active_session_count(), 2)

    # -----------------------------------------------------------------
    # 9. Repeated Attempts After Reset
    # -----------------------------------------------------------------
    def test_repeated_attempts_after_reset(self):
        """Verifies that calling reset() allows clean re-testing."""
        self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)
        self.assertEqual(self.detector.get_active_session_count(), 1)

        self.detector.reset()
        self.assertEqual(self.detector.get_active_session_count(), 0)

        # New attempt after reset works cleanly
        r_new = self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)
        self.assertEqual(self.detector.get_active_session_count(), 1)
        self.assertEqual(r_new.state, LivenessState.INSUFFICIENT_DATA)

    # -----------------------------------------------------------------
    # 10. Challenge Expiration
    # -----------------------------------------------------------------
    def test_challenge_verification_expires(self):
        """
        Verifies that a completed liveness verification expires after its validity duration,
        requiring the passenger to re-verify rather than remaining permanently valid.
        """
        # Pass challenge at t = 0
        self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now)
        self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now + timedelta(seconds=0.1))
        r_pass = self.detector.evaluate(self.box, make_landmarks("TURN_LEFT"), self.now + timedelta(seconds=0.3))
        self.assertTrue(r_pass.is_live)

        # Still valid within validity window (e.g. +1.0 second)
        r_within = self.detector.evaluate(self.box, make_landmarks("TURN_LEFT"), self.now + timedelta(seconds=1.3))
        self.assertTrue(r_within.is_live)

        # Expired after 4.0 seconds (validity is 3.0 seconds)
        r_expired = self.detector.evaluate(self.box, make_landmarks("NEUTRAL"), self.now + timedelta(seconds=4.5))
        self.assertFalse(r_expired.is_live)
        self.assertEqual(r_expired.state, LivenessState.INSUFFICIENT_DATA)

    # -----------------------------------------------------------------
    # 11. Passive Static Photo Detection (Zero Variance)
    # -----------------------------------------------------------------
    def test_passive_static_photo_check(self):
        """
        Verifies that presenting a completely static photograph with identical
        landmarks across frames triggers SPOOF_SUSPECTED.
        """
        pts_static = make_landmarks("NEUTRAL")
        for i in range(8):
            t = self.now + timedelta(seconds=0.25 * i)
            res = self.detector.evaluate(self.box, pts_static, t)

        # After 1.75 seconds of zero variance across 8 frames
        self.assertFalse(res.is_live)
        self.assertEqual(res.state, LivenessState.SPOOF_SUSPECTED)
        self.assertIn("motionless", res.reason.lower())


if __name__ == "__main__":
    unittest.main()
