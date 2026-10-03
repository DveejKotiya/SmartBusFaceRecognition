"""
End-to-End Pipeline Integration Tests for Stage 9
Smart Bus Face Recognition and Pass Verification System

Covers all 7 verification pipeline scenarios:
1. Registered live student -> ENTRY ALLOWED
2. Registered student using photo attack -> ENTRY DENIED (LIVENESS_FAILED)
3. Registered student with expired pass -> ENTRY DENIED (PASS_EXPIRED)
4. Registered student on wrong route -> ENTRY DENIED (ROUTE_MISMATCH)
5. Registered student rescanning within 5 minutes -> ENTRY DENIED (DUPLICATE_COOLDOWN)
6. Unknown person -> ENTRY DENIED (UNKNOWN_PERSON)
7. Multiple people in same frame -> independent processing without state leakage
"""

import sys
import tempfile
import sqlite3
import unittest
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.schema import (
    CREATE_STUDENTS_TABLE,
    CREATE_BUS_ROUTES_TABLE,
    CREATE_BUS_PASSES_TABLE,
    CREATE_ENTRY_LOGS_TABLE
)
from src.vision.face_detector import DetectionResult
from src.vision.face_recognizer import FaceRecognizer
from src.vision.embedding_store import EmbeddingStore
from src.vision.liveness import LivenessDetector, LivenessState, ChallengeType
from src.core.pass_verifier import PassVerifier, ReasonCode
from src.core.bus_entry_service import BusEntryService


def make_landmarks(yaw_type: str = "NEUTRAL") -> np.ndarray:
    pts = np.zeros((5, 2), dtype=np.float32)
    pts[0] = [60.0, 50.0]   # Left Eye
    pts[1] = [100.0, 50.0]  # Right Eye
    pts[3] = [65.0, 90.0]   # Mouth Left
    pts[4] = [95.0, 90.0]   # Mouth Right
    if yaw_type == "NEUTRAL":
        pts[2] = [80.0, 70.0]
    elif yaw_type == "TURN_LEFT":
        pts[2] = [68.0, 70.0]
    elif yaw_type == "TURN_RIGHT":
        pts[2] = [92.0, 70.0]
    return pts


class TestStage9Pipeline(unittest.TestCase):
    """Integration test suite covering the full 9-step boarding pipeline with liveness."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bus.db"
        self.store_path = Path(self.temp_dir.name) / "test_store.pkl"
        self._init_database()

        # Seed embedding store
        self.emb_aarav = np.zeros(512, dtype=np.float32)
        self.emb_aarav[0] = 1.0

        self.emb_priya = np.zeros(512, dtype=np.float32)
        self.emb_priya[1] = 1.0

        self.emb_rohan = np.zeros(512, dtype=np.float32)
        self.emb_rohan[2] = 1.0

        self.store = EmbeddingStore(storage_path=self.store_path)
        self.store.add_embedding("21CS101", "Aarav Sharma", self.emb_aarav)
        self.store.add_embedding("21EC202", "Priya Patel", self.emb_priya)
        self.store.add_embedding("21ME303", "Rohan Gupta", self.emb_rohan)
        self.store.save()

        self.verifier = PassVerifier(db_path=self.db_path, recognition_threshold=0.60, cooldown_seconds=300)
        self.mock_detector = MagicMock()
        self.mock_recognizer = MagicMock()

        self.liveness_detector = LivenessDetector(
            enabled=True,
            challenge_timeout=3.0,
            min_frames=2,
            challenge_type="TURN_LEFT"
        )

        self.service = BusEntryService(
            detector=self.mock_detector,
            recognizer=self.mock_recognizer,
            store=self.store,
            verifier=self.verifier,
            liveness_detector=self.liveness_detector,
            recognition_threshold=0.60,
            enable_liveness=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute(CREATE_STUDENTS_TABLE)
        cursor.execute(CREATE_BUS_ROUTES_TABLE)
        cursor.execute(CREATE_BUS_PASSES_TABLE)
        cursor.execute(CREATE_ENTRY_LOGS_TABLE)

        # Routes: BUS-12 (R-101), BUS-08 (R-102)
        cursor.execute("INSERT INTO bus_routes (id, route_code, route_name, bus_number) VALUES (1, 'R-101', 'Route 1', 'BUS-12');")
        cursor.execute("INSERT INTO bus_routes (id, route_code, route_name, bus_number) VALUES (2, 'R-102', 'Route 2', 'BUS-08');")

        # Students
        cursor.execute("INSERT INTO students (id, roll_number, name, department, is_active) VALUES (1, '21CS101', 'Aarav Sharma', 'CS', 1);")
        cursor.execute("INSERT INTO students (id, roll_number, name, department, is_active) VALUES (2, '21EC202', 'Priya Patel', 'EC', 1);")
        cursor.execute("INSERT INTO students (id, roll_number, name, department, is_active) VALUES (3, '21ME303', 'Rohan Gupta', 'ME', 1);")

        today = date.today().isoformat()
        future = (date.today() + timedelta(days=60)).isoformat()
        past_start = (date.today() - timedelta(days=120)).isoformat()
        past_end = (date.today() - timedelta(days=10)).isoformat()

        # Aarav: Active pass on Route 1 (R-101)
        cursor.execute("INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status) VALUES (1, 1, 'Semester', ?, ?, 'ACTIVE');", (today, future))
        # Priya: Expired pass on Route 1 (R-101)
        cursor.execute("INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status) VALUES (2, 1, 'Semester', ?, ?, 'ACTIVE');", (past_start, past_end))
        # Rohan: Active pass on Route 2 (R-102)
        cursor.execute("INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status) VALUES (3, 2, 'Semester', ?, ?, 'ACTIVE');", (today, future))

        conn.commit()
        conn.close()

    def _dummy_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Registered Live Student -> ENTRY ALLOWED
    def test_scenario_1_live_student_allowed(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # Frame 1: Neutral baseline
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        r1 = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))
        self.assertFalse(r1[0].allowed)
        self.assertEqual(r1[0].reason_code, ReasonCode.LIVENESS_INCONCLUSIVE)

        # Frame 2: Turn Left response
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        r2 = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))

        self.assertTrue(r2[0].allowed)
        self.assertEqual(r2[0].student_id, "21CS101")
        self.assertEqual(r2[0].student_name, "Aarav Sharma")
        self.assertEqual(r2[0].reason_code, ReasonCode.PASS_VALID)

    # 2. Registered Student Using Photo -> ENTRY DENIED
    def test_scenario_2_photo_attack_denied(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # Initial frame
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))

        # Photo remains motionless and times out at t = +3.5s
        r_timeout = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 4, tzinfo=timezone.utc))

        self.assertFalse(r_timeout[0].allowed)
        self.assertEqual(r_timeout[0].reason_code, ReasonCode.LIVENESS_FAILED)
        self.assertIn("Liveness verification failed", r_timeout[0].reason_message)

    # 3. Registered Student with Expired Pass -> ENTRY DENIED
    def test_scenario_3_expired_pass_denied(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_priya

        # Pass liveness
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))

        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        r = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))

        self.assertFalse(r[0].allowed)
        self.assertEqual(r[0].student_id, "21EC202")
        self.assertEqual(r[0].reason_code, ReasonCode.PASS_EXPIRED)

    # 4. Registered Student on Wrong Route -> ENTRY DENIED
    def test_scenario_4_wrong_route_denied(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_rohan  # Rohan is on R-102

        # Pass liveness
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))

        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        # Attempt boarding on R-101
        r = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))

        self.assertFalse(r[0].allowed)
        self.assertEqual(r[0].student_id, "21ME303")
        self.assertEqual(r[0].reason_code, ReasonCode.ROUTE_MISMATCH)

    # 5. Duplicate Boarding Cooldown -> ENTRY DENIED
    def test_scenario_5_duplicate_cooldown_denied(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # Boarding 1: Allowed at 08:00
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        r1 = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))
        self.assertTrue(r1[0].allowed)

        # Boarding 2: Attempted 60 seconds later (within 300s cooldown)
        t_soon = datetime(2026, 10, 3, 8, 1, 0, tzinfo=timezone.utc)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=t_soon)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        r2 = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=t_soon + timedelta(seconds=1))

        self.assertFalse(r2[0].allowed)
        self.assertEqual(r2[0].reason_code, ReasonCode.DUPLICATE_COOLDOWN)
        self.assertTrue(r2[0].duplicate)

    # 6. Unknown Person -> ENTRY DENIED
    def test_scenario_6_unknown_person_denied(self):
        box = (50, 50, 200, 200)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        # Unrecognized face embedding (orthogonal to Aarav, Priya, Rohan)
        emb_unknown = np.zeros(512, dtype=np.float32)
        emb_unknown[10] = 1.0
        self.mock_recognizer.generate_embedding.return_value = emb_unknown

        # Pass liveness
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))

        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT"))
        ]
        r = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))

        self.assertFalse(r[0].allowed)
        self.assertEqual(r[0].reason_code, ReasonCode.UNKNOWN_PERSON)
        self.assertIsNone(r[0].student_id)

    # 7. Multiple People -> Independent Processing
    def test_scenario_7_multiple_people_independent(self):
        box_live = (50, 50, 150, 150)
        box_spoof = (400, 50, 500, 150)
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # Frame 1: Both neutral
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box_live, confidence=0.98, face_crop=crop, landmarks=make_landmarks("NEUTRAL")),
            DetectionResult(box=box_spoof, confidence=0.95, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc))

        # Frame 2: Person 1 turns left, Person 2 remains motionless (spoof)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=box_live, confidence=0.98, face_crop=crop, landmarks=make_landmarks("TURN_LEFT")),
            DetectionResult(box=box_spoof, confidence=0.95, face_crop=crop, landmarks=make_landmarks("NEUTRAL"))
        ]
        results = self.service.process_frame(self._dummy_frame(), "BUS-12", "R-101", current_timestamp=datetime(2026, 10, 3, 8, 0, 1, tzinfo=timezone.utc))

        self.assertEqual(len(results), 2)
        # Person 1 is verified and allowed
        self.assertTrue(results[0].allowed)
        self.assertEqual(results[0].student_id, "21CS101")

        # Person 2 has not satisfied challenge, so NOT allowed
        self.assertFalse(results[1].allowed)
        self.assertEqual(results[1].reason_code, ReasonCode.LIVENESS_INCONCLUSIVE)


if __name__ == "__main__":
    unittest.main()
