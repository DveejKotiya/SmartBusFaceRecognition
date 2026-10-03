"""
Unit Tests for PassVerifier Business Rules Engine
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
from datetime import datetime, date, timedelta, timezone

from src.core.pass_verifier import (
    PassVerifier,
    RecognitionResult,
    VerificationResult,
    ReasonCode,
    DEFAULT_RECOGNITION_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS
)
from src.database.schema import (
    CREATE_STUDENTS_TABLE,
    CREATE_BUS_ROUTES_TABLE,
    CREATE_BUS_PASSES_TABLE,
    CREATE_ENTRY_LOGS_TABLE
)


class TestPassVerifier(unittest.TestCase):
    """Deterministic test suite for PassVerifier rules and cooldown logic."""

    def setUp(self):
        # Create a fresh temporary SQLite database for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bus.db"
        self._init_test_database()

        # Initialize verifier pointing to the isolated test database
        self.verifier = PassVerifier(
            db_path=self.db_path,
            recognition_threshold=0.60,
            cooldown_seconds=300
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_test_database(self):
        """Initializes tables and seeds test fixtures."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute(CREATE_STUDENTS_TABLE)
        cursor.execute(CREATE_BUS_ROUTES_TABLE)
        cursor.execute(CREATE_BUS_PASSES_TABLE)
        cursor.execute(CREATE_ENTRY_LOGS_TABLE)

        # Seed Route 101 (BUS-12) and Route 102 (BUS-08)
        cursor.execute("""
            INSERT INTO bus_routes (id, route_code, route_name, bus_number, driver_name)
            VALUES 
            (1, 'R-101', 'Hostel to Campus', 'BUS-12', 'Driver One'),
            (2, 'R-102', 'Station to Campus', 'BUS-08', 'Driver Two');
        """)

        # Seed Students
        cursor.execute("""
            INSERT INTO students (id, roll_number, name, department, is_active)
            VALUES 
            (1, '21CS101', 'Aarav Sharma', 'CS', 1),
            (2, '21EC202', 'Priya Patel', 'EC', 1),
            (3, '21ME303', 'Rohan Gupta', 'ME', 1),
            (4, '21CV404', 'Inactive Student', 'CV', 0),
            (5, '21IT505', 'No Pass Student', 'IT', 1);
        """)

        # Current reference date
        today = date.today()
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        next_month = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        last_month = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        last_year = (today - timedelta(days=365)).strftime("%Y-%m-%d")

        # Passes:
        # Student 1: Valid active pass on Route 1 (R-101)
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (1, 1, 'Semester', ?, ?, 'ACTIVE');
        """, (last_month, next_month))

        # Student 2: Expired pass on Route 1
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (2, 1, 'Semester', ?, ?, 'ACTIVE');
        """, (last_year, yesterday))

        # Student 3: Future pass (not started yet) on Route 1
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (3, 1, 'Semester', ?, ?, 'ACTIVE');
        """, (tomorrow, next_month))

        # Student 4: Inactive status pass on Route 1
        cursor.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (4, 1, 'Semester', ?, ?, 'REVOKED');
        """, (last_month, next_month))

        conn.commit()
        conn.close()

    # -------------------------------------------------------------
    # 1. Unknown Face Check
    # -------------------------------------------------------------
    def test_unknown_face_rejected(self):
        rec = RecognitionResult(recognized=False, student_id=None, similarity=0.2)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.UNKNOWN_PERSON)

    # -------------------------------------------------------------
    # 2. Student Not in Database
    # -------------------------------------------------------------
    def test_student_not_found(self):
        rec = RecognitionResult(recognized=True, student_id="99GHOST", student_name="Ghost", similarity=0.92)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.STUDENT_NOT_FOUND)

    # -------------------------------------------------------------
    # 3. No Pass Issued
    # -------------------------------------------------------------
    def test_no_pass_issued(self):
        rec = RecognitionResult(recognized=True, student_id="21IT505", student_name="No Pass Student", similarity=0.88)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.NO_PASS)

    # -------------------------------------------------------------
    # 4. Inactive / Revoked Pass
    # -------------------------------------------------------------
    def test_inactive_pass(self):
        # We manually insert an inactive pass for an active student
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (5, 1, 'Semester', '2026-01-01', '2026-12-31', 'SUSPENDED');
        """)
        conn.commit()
        conn.close()

        rec = RecognitionResult(recognized=True, student_id="21IT505", similarity=0.85)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.PASS_INACTIVE)

    # -------------------------------------------------------------
    # 5. Pass Not Started Yet
    # -------------------------------------------------------------
    def test_pass_not_started_yet(self):
        rec = RecognitionResult(recognized=True, student_id="21ME303", similarity=0.89)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.PASS_NOT_STARTED)

    # -------------------------------------------------------------
    # 6. Expired Pass
    # -------------------------------------------------------------
    def test_pass_expired(self):
        rec = RecognitionResult(recognized=True, student_id="21EC202", similarity=0.91)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.PASS_EXPIRED)

    # -------------------------------------------------------------
    # 7. Valid Pass & 8. Correct Route
    # -------------------------------------------------------------
    def test_valid_pass_allowed(self):
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.95)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertTrue(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.PASS_VALID)
        self.assertEqual(res.student_name, "Aarav Sharma")

    # -------------------------------------------------------------
    # 9. Wrong Route Mismatch
    # -------------------------------------------------------------
    def test_wrong_route_mismatch(self):
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.95)
        # Student 1 has pass on R-101 (BUS-12), but boards R-102 (BUS-08)
        res = self.verifier.verify(rec, bus_id="BUS-08", route="R-102")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.ROUTE_MISMATCH)

    # -------------------------------------------------------------
    # 10. Similarity Below Threshold
    # -------------------------------------------------------------
    def test_similarity_below_threshold(self):
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.55)  # threshold is 0.60
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.LOW_CONFIDENCE)

    # -------------------------------------------------------------
    # 11. Similarity Above Threshold
    # -------------------------------------------------------------
    def test_similarity_above_threshold(self):
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.60)  # exactly threshold
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertTrue(res.allowed)

    # -------------------------------------------------------------
    # 12. First Boarding Allowed
    # -------------------------------------------------------------
    def test_first_boarding_allowed(self):
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.88)
        res = self.verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertTrue(res.allowed)
        self.assertFalse(res.duplicate)

    # -------------------------------------------------------------
    # 13. Same Student + Same Bus Within 5 Minutes Rejected (Cooldown)
    # -------------------------------------------------------------
    def test_cooldown_within_5_minutes_rejected(self):
        t0 = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)

        # 1. First scan at 10:00:00 -> Allowed and logged
        res1 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t0)
        self.assertTrue(res1.allowed)
        self.verifier.log_verification(res1)

        # 2. Duplicate scan at 10:02:00 (120s later) -> Rejected by cooldown
        t1 = datetime(2026, 10, 3, 10, 2, 0, tzinfo=timezone.utc)
        res2 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t1)
        self.assertFalse(res2.allowed)
        self.assertTrue(res2.duplicate)
        self.assertEqual(res2.reason_code, ReasonCode.DUPLICATE_COOLDOWN)

    # -------------------------------------------------------------
    # 14. Same Student + Same Bus After 5 Minutes Allowed
    # -------------------------------------------------------------
    def test_cooldown_after_5_minutes_allowed(self):
        t0 = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)

        # First scan
        res1 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t0)
        self.verifier.log_verification(res1)

        # Scan 6 minutes later (360 seconds > 300)
        t_after = datetime(2026, 10, 3, 10, 6, 0, tzinfo=timezone.utc)
        res2 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t_after)
        self.assertTrue(res2.allowed)
        self.assertFalse(res2.duplicate)
        self.assertEqual(res2.reason_code, ReasonCode.PASS_VALID)

    # -------------------------------------------------------------
    # 15. Same Student + Different Bus Not Blocked by Other Bus
    # -------------------------------------------------------------
    def test_different_bus_independent_cooldown(self):
        # Student 1 boards Bus 1 (Route 1) at 10:00:00
        t0 = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)
        res1 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t0)
        self.verifier.log_verification(res1)

        # Give Student 1 a pass on Route 2 as well
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (1, 2, 'Semester', '2026-01-01', '2026-12-31', 'ACTIVE');
        """)
        conn.commit()
        conn.close()

        # At 10:02:00, student tries Route 2 (different route ID) -> cooldown should not block Route 2
        t1 = datetime(2026, 10, 3, 10, 2, 0, tzinfo=timezone.utc)
        res2 = self.verifier.verify(rec, bus_id="BUS-08", route="R-102", current_timestamp=t1)
        self.assertTrue(res2.allowed)
        self.assertFalse(res2.duplicate)

    # -------------------------------------------------------------
    # 16. Different Student + Same Bus Allowed Immediately
    # -------------------------------------------------------------
    def test_different_student_same_bus_allowed(self):
        t0 = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)

        # Student 1 boards
        rec1 = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)
        res1 = self.verifier.verify(rec1, bus_id="BUS-12", route="R-101", current_timestamp=t0)
        self.verifier.log_verification(res1)

        # Add valid pass for Student 5 on Route 1
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (5, 1, 'Semester', '2026-01-01', '2026-12-31', 'ACTIVE');
        """)
        conn.commit()
        conn.close()

        # Student 5 boards on same bus at 10:00:10 (10 seconds later)
        t_sec = datetime(2026, 10, 3, 10, 0, 10, tzinfo=timezone.utc)
        rec2 = RecognitionResult(recognized=True, student_id="21IT505", similarity=0.88)
        res2 = self.verifier.verify(rec2, bus_id="BUS-12", route="R-101", current_timestamp=t_sec)
        self.assertTrue(res2.allowed)
        self.assertFalse(res2.duplicate)

    # -------------------------------------------------------------
    # 17. Database Restart Simulation for Cooldown Persistence
    # -------------------------------------------------------------
    def test_cooldown_persists_across_verifier_instances(self):
        t0 = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)

        # Log boarding with Verifier instance A
        res1 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t0)
        self.verifier.log_verification(res1)

        # Simulate system restart by destroying verifier instance and creating Verifier instance B
        restarted_verifier = PassVerifier(
            db_path=self.db_path,
            recognition_threshold=0.60,
            cooldown_seconds=300
        )

        # Check boarding 2 minutes later with new instance
        t1 = datetime(2026, 10, 3, 10, 2, 0, tzinfo=timezone.utc)
        res2 = restarted_verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t1)
        self.assertFalse(res2.allowed)
        self.assertTrue(res2.duplicate)
        self.assertEqual(res2.reason_code, ReasonCode.DUPLICATE_COOLDOWN)

    # -------------------------------------------------------------
    # 18. Missing / Corrupted Database Handled Safely
    # -------------------------------------------------------------
    def test_missing_database_file_handled(self):
        bad_verifier = PassVerifier(db_path=Path("Z:/nonexistent_dir/bad.db"))
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)
        res = bad_verifier.verify(rec, bus_id="BUS-12", route="R-101")
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.DATABASE_ERROR)

    # -------------------------------------------------------------
    # 19. Exact Cooldown Boundary Testing: 10:04:59, 10:05:00, 10:05:01
    # -------------------------------------------------------------
    def test_cooldown_exact_boundaries(self):
        """
        Convention:
        Cooldown window = 300 seconds.
        If elapsed < 300 seconds -> REJECTED (DUPLICATE_COOLDOWN)
        If elapsed >= 300 seconds -> ALLOWED (Cooldown elapsed)

        Initial boarding at 10:00:00:
        10:04:59 (elapsed = 299s) -> DENIED
        10:05:00 (elapsed = 300s) -> ALLOWED
        10:05:01 (elapsed = 301s) -> ALLOWED
        """
        t_base = datetime(2026, 10, 3, 10, 0, 0, tzinfo=timezone.utc)
        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)

        # Initial entry
        res0 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t_base)
        self.assertTrue(res0.allowed)
        self.verifier.log_verification(res0)

        # Boundary 1: 10:04:59 (299 seconds elapsed) -> Must be DENIED
        t_299 = datetime(2026, 10, 3, 10, 4, 59, tzinfo=timezone.utc)
        res_299 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t_299)
        self.assertFalse(res_299.allowed, "299s elapsed must trigger cooldown rejection")
        self.assertEqual(res_299.reason_code, ReasonCode.DUPLICATE_COOLDOWN)

        # Boundary 2: 10:05:00 (300 seconds elapsed) -> Must be ALLOWED
        t_300 = datetime(2026, 10, 3, 10, 5, 0, tzinfo=timezone.utc)
        res_300 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t_300)
        self.assertTrue(res_300.allowed, "300s elapsed must clear cooldown")
        self.assertEqual(res_300.reason_code, ReasonCode.PASS_VALID)

        # Boundary 3: 10:05:01 (301 seconds elapsed) -> Must be ALLOWED
        t_301 = datetime(2026, 10, 3, 10, 5, 1, tzinfo=timezone.utc)
        res_301 = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=t_301)
        self.assertTrue(res_301.allowed, "301s elapsed must clear cooldown")
        self.assertEqual(res_301.reason_code, ReasonCode.PASS_VALID)

    # -------------------------------------------------------------
    # 20. Timezone & Inclusive Date Boundary Cases
    # -------------------------------------------------------------
    def test_inclusive_expiration_and_timezone_handling(self):
        """
        Verify that a pass expiring on 2026-10-03 is valid throughout all of 2026-10-03,
        and expires on 2026-10-04.
        """
        # Create pass expiring exactly on 2026-10-03
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
            VALUES (1, 1, 'Day', '2026-10-03', '2026-10-03', 'ACTIVE');
        """)
        conn.commit()
        conn.close()

        rec = RecognitionResult(recognized=True, student_id="21CS101", similarity=0.90)

        # Test at 2026-10-03 23:59:59 UTC -> Valid (Inclusive)
        dt_eod = datetime(2026, 10, 3, 23, 59, 59, tzinfo=timezone.utc)
        res_eod = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=dt_eod)
        self.assertTrue(res_eod.allowed, "Pass must be valid on expiry date until end of day")
        self.assertEqual(res_eod.reason_code, ReasonCode.PASS_VALID)

        # Test next morning 2026-10-04 00:00:01 UTC -> Expired
        dt_next_day = datetime(2026, 10, 4, 0, 0, 1, tzinfo=timezone.utc)
        res_next_day = self.verifier.verify(rec, bus_id="BUS-12", route="R-101", current_timestamp=dt_next_day)
        self.assertFalse(res_next_day.allowed, "Pass must expire on day after expiry_date")
        self.assertEqual(res_next_day.reason_code, ReasonCode.PASS_EXPIRED)


if __name__ == "__main__":
    unittest.main()
