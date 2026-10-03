"""
Unit Tests for BusEntryService
Smart Bus Face Recognition and Pass Verification System
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import tempfile
import sqlite3
from unittest.mock import MagicMock
from datetime import datetime, date, timedelta, timezone
import numpy as np

from src.core.bus_entry_service import BusEntryService, BusEntryResult
from src.core.pass_verifier import PassVerifier, ReasonCode
from src.vision.face_detector import DetectionResult
from src.vision.embedding_store import EmbeddingStore
from src.database.schema import (
    CREATE_STUDENTS_TABLE,
    CREATE_BUS_ROUTES_TABLE,
    CREATE_BUS_PASSES_TABLE,
    CREATE_ENTRY_LOGS_TABLE
)


class TestBusEntryService(unittest.TestCase):
    """Test suite covering the orchestration service without requiring webcam hardware."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bus.db"
        self.store_path = Path(self.temp_dir.name) / "test_store.pkl"
        self._init_test_database()

        # Deterministic 512-D test embeddings
        self.emb_aarav = np.zeros(512, dtype=np.float32)
        self.emb_aarav[0] = 1.0  # Unit vector for Aarav

        self.emb_priya = np.zeros(512, dtype=np.float32)
        self.emb_priya[1] = 1.0  # Unit vector for Priya

        self.emb_rohan = np.zeros(512, dtype=np.float32)
        self.emb_rohan[2] = 1.0  # Unit vector for Rohan

        # Populate EmbeddingStore
        self.store = EmbeddingStore(storage_path=self.store_path)
        self.store.add_embedding("21CS101", "Aarav Sharma", self.emb_aarav)
        self.store.add_embedding("21EC202", "Priya Patel", self.emb_priya)
        self.store.add_embedding("21ME303", "Rohan Gupta", self.emb_rohan)
        self.store.save()

        # PassVerifier
        self.verifier = PassVerifier(
            db_path=self.db_path,
            recognition_threshold=0.60,
            cooldown_seconds=300
        )

        # Mock FaceDetector and FaceRecognizer
        self.mock_detector = MagicMock()
        self.mock_recognizer = MagicMock()

        self.service = BusEntryService(
            detector=self.mock_detector,
            recognizer=self.mock_recognizer,
            store=self.store,
            verifier=self.verifier,
            recognition_threshold=0.60
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_test_database(self):
        """Creates tables and seeds test records."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute(CREATE_STUDENTS_TABLE)
        cursor.execute(CREATE_BUS_ROUTES_TABLE)
        cursor.execute(CREATE_BUS_PASSES_TABLE)
        cursor.execute(CREATE_ENTRY_LOGS_TABLE)

        # Routes
        cursor.execute("""
            INSERT INTO bus_routes (id, route_code, route_name, bus_number)
            VALUES 
            (1, 'R-101', 'Route 101', 'BUS-12'),
            (2, 'R-102', 'Route 102', 'BUS-08');
        """)

        # Students
        cursor.execute("""
            INSERT INTO students (id, roll_number, name, department, is_active)
            VALUES 
            (1, '21CS101', 'Aarav Sharma', 'CS', 1),
            (2, '21EC202', 'Priya Patel', 'EC', 1),
            (3, '21ME303', 'Rohan Gupta', 'ME', 1);
        """)

        today = date.today()
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        next_month = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        last_year = (today - timedelta(days=365)).strftime("%Y-%m-%d")

        # Passes:
        # Aarav: Active pass on Route 1 (R-101)
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (1, 1, 'Semester', ?, ?, 'ACTIVE');
        """, (last_year, next_month))

        # Priya: Expired pass on Route 1
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (2, 1, 'Semester', ?, ?, 'ACTIVE');
        """, (last_year, yesterday))

        # Rohan: Active pass on Route 2 (R-102)
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (3, 2, 'Semester', ?, ?, 'ACTIVE');
        """, (last_year, next_month))

        conn.commit()
        conn.close()

    def _dummy_frame(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # -------------------------------------------------------------
    # 1. Known Student + Valid Pass
    # -------------------------------------------------------------
    def test_known_student_valid_pass_allowed(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.98, face_crop=dummy_crop)
        ]
        # Recognizer returns Aarav's embedding
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101",
            log_to_db=True
        )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r.allowed)
        self.assertEqual(r.student_id, "21CS101")
        self.assertEqual(r.student_name, "Aarav Sharma")
        self.assertAlmostEqual(r.similarity, 1.0, places=3)
        self.assertEqual(r.reason_code, ReasonCode.PASS_VALID)

    # -------------------------------------------------------------
    # 2. Known Student + Expired Pass
    # -------------------------------------------------------------
    def test_known_student_expired_pass_denied(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.95, face_crop=dummy_crop)
        ]
        # Priya's embedding
        self.mock_recognizer.generate_embedding.return_value = self.emb_priya

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertFalse(r.allowed)
        self.assertEqual(r.student_id, "21EC202")
        self.assertEqual(r.reason_code, ReasonCode.PASS_EXPIRED)

    # -------------------------------------------------------------
    # 3. Known Student + Wrong Route
    # -------------------------------------------------------------
    def test_known_student_wrong_route_denied(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.95, face_crop=dummy_crop)
        ]
        # Rohan's embedding (assigned to R-102) boards R-101
        self.mock_recognizer.generate_embedding.return_value = self.emb_rohan

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertFalse(r.allowed)
        self.assertEqual(r.student_id, "21ME303")
        self.assertEqual(r.reason_code, ReasonCode.ROUTE_MISMATCH)

    # -------------------------------------------------------------
    # 4. Unknown Person
    # -------------------------------------------------------------
    def test_unknown_person_denied(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.92, face_crop=dummy_crop)
        ]
        # Unknown vector orthogonal to Aarav, Priya, and Rohan
        unknown_emb = np.zeros(512, dtype=np.float32)
        unknown_emb[10] = 1.0
        self.mock_recognizer.generate_embedding.return_value = unknown_emb

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertFalse(r.allowed)
        self.assertIsNone(r.student_id)
        self.assertIsNone(r.student_name)
        self.assertEqual(r.reason_code, ReasonCode.UNKNOWN_PERSON)

    # -------------------------------------------------------------
    # 5. Duplicate Cooldown
    # -------------------------------------------------------------
    def test_duplicate_cooldown_denied(self):
        t0 = datetime(2026, 10, 3, 8, 0, 0, tzinfo=timezone.utc)
        t_soon = datetime(2026, 10, 3, 8, 2, 0, tzinfo=timezone.utc)  # 2 mins later

        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.98, face_crop=dummy_crop)
        ]
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # 1. First scan at 08:00 -> Allowed
        res1 = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101",
            current_timestamp=t0,
            log_to_db=True
        )
        self.assertTrue(res1[0].allowed)

        # 2. Second scan at 08:02 -> Denied due to cooldown
        res2 = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101",
            current_timestamp=t_soon,
            log_to_db=True
        )
        self.assertFalse(res2[0].allowed)
        self.assertTrue(res2[0].duplicate)
        self.assertEqual(res2[0].reason_code, ReasonCode.DUPLICATE_COOLDOWN)

    # -------------------------------------------------------------
    # 6. Multiple Faces in Single Frame
    # -------------------------------------------------------------
    def test_multiple_faces_processed_independently(self):
        crop1 = np.zeros((160, 160, 3), dtype=np.uint8)
        crop2 = np.ones((160, 160, 3), dtype=np.uint8)

        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(20, 20, 100, 100), confidence=0.95, face_crop=crop1),
            DetectionResult(box=(200, 20, 280, 100), confidence=0.90, face_crop=crop2)
        ]
        # Side effect: first call returns Aarav, second call returns Priya (expired)
        self.mock_recognizer.generate_embedding.side_effect = [self.emb_aarav, self.emb_priya]

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )

        self.assertEqual(len(results), 2)
        # Face 1: Aarav -> Allowed
        self.assertTrue(results[0].allowed)
        self.assertEqual(results[0].student_id, "21CS101")

        # Face 2: Priya -> Denied (Expired)
        self.assertFalse(results[1].allowed)
        self.assertEqual(results[1].student_id, "21EC202")

    # -------------------------------------------------------------
    # 7. No Faces in Frame
    # -------------------------------------------------------------
    def test_no_faces_in_frame(self):
        self.mock_detector.detect_faces.return_value = []
        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )
        self.assertEqual(results, [])

    # -------------------------------------------------------------
    # 8. Invalid Frame (None or Empty)
    # -------------------------------------------------------------
    def test_invalid_frame_handled_safely(self):
        self.assertEqual(self.service.process_frame(None, "BUS-12", "R-101"), [])
        self.assertEqual(self.service.process_frame(np.array([]), "BUS-12", "R-101"), [])

    # -------------------------------------------------------------
    # 9. Recognition Failure (Low Confidence Below Threshold)
    # -------------------------------------------------------------
    def test_low_confidence_match_treated_as_unknown(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.88, face_crop=dummy_crop)
        ]
        # Create vector with similarity = 0.50 (below 0.60 threshold)
        v_low = (self.emb_aarav * 0.50) + (np.roll(self.emb_aarav, 100) * 0.866)
        v_low = v_low / np.linalg.norm(v_low)
        self.mock_recognizer.generate_embedding.return_value = v_low

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertFalse(r.allowed)
        self.assertIsNone(r.student_id)
        self.assertEqual(r.reason_code, ReasonCode.UNKNOWN_PERSON)

    # -------------------------------------------------------------
    # 10. Database Failure Handled Gracefully
    # -------------------------------------------------------------
    def test_database_failure_does_not_crash_service(self):
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        self.mock_detector.detect_faces.return_value = [
            DetectionResult(box=(50, 50, 200, 200), confidence=0.98, face_crop=dummy_crop)
        ]
        self.mock_recognizer.generate_embedding.return_value = self.emb_aarav

        # Force database exception in verifier
        self.service.verifier._get_connection = MagicMock(side_effect=sqlite3.OperationalError("Database disk full"))

        results = self.service.process_frame(
            frame=self._dummy_frame(),
            bus_id="BUS-12",
            route="R-101"
        )
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].allowed)
        self.assertEqual(results[0].reason_code, ReasonCode.DATABASE_ERROR)


if __name__ == "__main__":
    unittest.main()
