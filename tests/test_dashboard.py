"""
Unit Tests for Dashboard and Reporting Backend
Smart Bus Face Recognition and Pass Verification System

Covers:
1. Dashboard summary statistics calculation
2. Log filtering (by date, student, bus, route, result)
3. CSV export formatting and expected headers (no raw biometrics)
4. Student search and filter logic
5. Bus pass status filter logic
6. System status diagnostic functions
"""

import sys
import io
import csv
import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
import pandas as pd

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
from src.database.dashboard_queries import (
    get_dashboard_summary,
    get_recent_activity,
    get_students_directory,
    get_passes_directory,
    get_filtered_entry_logs,
    get_distinct_buses,
    get_distinct_routes,
    get_all_departments,
    get_all_semesters
)
import src.ui.system_page as sp


class TestDashboardQueriesAndExports(unittest.TestCase):
    """Test suite covering dashboard aggregation, search, filters, and reports."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_smart_bus.db"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute(CREATE_STUDENTS_TABLE)
            cursor.execute(CREATE_BUS_ROUTES_TABLE)
            cursor.execute(CREATE_BUS_PASSES_TABLE)
            cursor.execute(CREATE_ENTRY_LOGS_TABLE)
            cursor.execute("ALTER TABLE students ADD COLUMN semester TEXT;")

            # 1. Insert Routes
            cursor.execute("INSERT INTO bus_routes (bus_number, route_code, route_name) VALUES ('BUS-12', 'R-101', 'North Campus')")
            route_id_101 = cursor.lastrowid
            cursor.execute("INSERT INTO bus_routes (bus_number, route_code, route_name) VALUES ('BUS-08', 'R-102', 'South City')")
            route_id_102 = cursor.lastrowid

            # 2. Insert Students
            cursor.execute("INSERT INTO students (roll_number, name, department, semester) VALUES ('21CS101', 'Aarav Sharma', 'Computer Science', '6')")
            s1_id = cursor.lastrowid
            cursor.execute("INSERT INTO students (roll_number, name, department, semester) VALUES ('21EC202', 'Priya Patel', 'Electronics', '6')")
            s2_id = cursor.lastrowid
            cursor.execute("INSERT INTO students (roll_number, name, department, semester) VALUES ('21ME303', 'Rohan Gupta', 'Mechanical', '4')")
            s3_id = cursor.lastrowid

            # 3. Insert Bus Passes
            today = date.today().isoformat()
            future = (date.today() + timedelta(days=60)).isoformat()
            past_start = (date.today() - timedelta(days=120)).isoformat()
            past_end = (date.today() - timedelta(days=10)).isoformat()

            cursor.execute("""
                INSERT INTO bus_passes (student_id, route_id, start_date, expiry_date, status)
                VALUES (?, ?, ?, ?, 'ACTIVE')
            """, (s1_id, route_id_101, today, future))

            cursor.execute("""
                INSERT INTO bus_passes (student_id, route_id, start_date, expiry_date, status)
                VALUES (?, ?, ?, ?, 'EXPIRED')
            """, (s2_id, route_id_101, past_start, past_end))

            cursor.execute("""
                INSERT INTO bus_passes (student_id, route_id, start_date, expiry_date, status)
                VALUES (?, ?, ?, ?, 'INACTIVE')
            """, (s3_id, route_id_102, today, future))

            # 4. Insert Entry Logs in chronological order
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            yesterday_iso = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

            # 1. Approved yesterday (Aarav)
            cursor.execute("""
                INSERT INTO entry_logs (student_id, route_id, status, similarity_score, rejection_reason, timestamp, verification_method)
                VALUES (?, ?, 'APPROVED', 0.87, NULL, ?, 'FACE')
            """, (s1_id, route_id_101, yesterday_iso))

            # 2. Approved today (Aarav)
            cursor.execute("""
                INSERT INTO entry_logs (student_id, route_id, status, similarity_score, rejection_reason, timestamp, verification_method)
                VALUES (?, ?, 'APPROVED', 0.88, NULL, ?, 'FACE')
            """, (s1_id, route_id_101, now_iso))

            # 3. Rejected today - expired pass (Priya)
            cursor.execute("""
                INSERT INTO entry_logs (student_id, route_id, status, similarity_score, rejection_reason, timestamp, verification_method)
                VALUES (?, ?, 'REJECTED', 0.85, 'PASS_EXPIRED', ?, 'FACE')
            """, (s2_id, route_id_101, now_iso))

            # 4. Rejected today - unknown person (latest)
            cursor.execute("""
                INSERT INTO entry_logs (student_id, route_id, status, similarity_score, rejection_reason, timestamp, verification_method)
                VALUES (NULL, ?, 'REJECTED', 0.32, 'UNKNOWN_PERSON', ?, 'FACE')
            """, (route_id_101, now_iso))

            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_dashboard_summary_calculation(self):
        """Verifies that get_dashboard_summary accurately aggregates real database counts."""
        stats = get_dashboard_summary(db_path=self.db_path)

        self.assertEqual(stats["total_students"], 3)
        self.assertEqual(stats["active_passes"], 1)
        self.assertGreaterEqual(stats["expired_passes"], 1)
        self.assertEqual(stats["today_allowed"], 1)
        self.assertEqual(stats["today_denied"], 2)  # 1 expired pass + 1 unknown person
        self.assertEqual(stats["today_unknown"], 1)

    def test_recent_activity_feed(self):
        """Verifies that recent activity retrieves logs in reverse chronological order."""
        activity = get_recent_activity(limit=5, db_path=self.db_path)
        self.assertEqual(len(activity), 4)
        # Latest record should be the unknown person attempt
        self.assertEqual(activity[0]["result"], "REJECTED")
        self.assertEqual(activity[0]["reason"], "UNKNOWN_PERSON")
        self.assertEqual(activity[0]["student_name"], "UNKNOWN PERSON")
        self.assertEqual(activity[0]["student_id"], "N/A")

    def test_student_directory_search_and_filters(self):
        """Verifies student search by roll number, name, department, and semester."""
        # 1. Search by exact roll number
        res_roll = get_students_directory(search="21CS101", db_path=self.db_path)
        self.assertEqual(len(res_roll), 1)
        self.assertEqual(res_roll[0]["name"], "Aarav Sharma")

        # 2. Search by partial name
        res_name = get_students_directory(search="Priya", db_path=self.db_path)
        self.assertEqual(len(res_name), 1)
        self.assertEqual(res_name[0]["student_id"], "21EC202")

        # 3. Filter by department
        res_dept = get_students_directory(department="Mechanical", db_path=self.db_path)
        self.assertEqual(len(res_dept), 1)
        self.assertEqual(res_dept[0]["student_id"], "21ME303")

        # 4. Filter by semester
        res_sem = get_students_directory(semester="6", db_path=self.db_path)
        self.assertEqual(len(res_sem), 2)

        # 5. Non-existent student
        res_none = get_students_directory(search="NONEXISTENT", db_path=self.db_path)
        self.assertEqual(len(res_none), 0)

    def test_passes_directory_filters(self):
        """Verifies bus passes status filtering for ALL, ACTIVE, and EXPIRED."""
        all_passes = get_passes_directory(status_filter="ALL", db_path=self.db_path)
        self.assertEqual(len(all_passes), 3)

        active_passes = get_passes_directory(status_filter="ACTIVE", db_path=self.db_path)
        self.assertEqual(len(active_passes), 1)
        self.assertEqual(active_passes[0]["student_id"], "21CS101")

        expired_passes = get_passes_directory(status_filter="EXPIRED", db_path=self.db_path)
        self.assertEqual(len(expired_passes), 1)
        self.assertEqual(expired_passes[0]["student_id"], "21EC202")

    def test_filtered_entry_logs(self):
        """Verifies multi-attribute filtering on audit entry logs."""
        # 1. Filter by student query
        res_student = get_filtered_entry_logs(student_query="Aarav", db_path=self.db_path)
        self.assertEqual(len(res_student), 2)  # 1 today, 1 yesterday

        # 2. Filter by result = APPROVED
        res_approved = get_filtered_entry_logs(result="APPROVED", db_path=self.db_path)
        self.assertEqual(len(res_approved), 2)

        # 3. Filter by result = REJECTED
        res_rejected = get_filtered_entry_logs(result="REJECTED", db_path=self.db_path)
        self.assertEqual(len(res_rejected), 2)

        # 4. Filter by result = UNKNOWN
        res_unknown = get_filtered_entry_logs(result="UNKNOWN", db_path=self.db_path)
        self.assertEqual(len(res_unknown), 1)
        self.assertEqual(res_unknown[0]["reason"], "UNKNOWN_PERSON")

        # 5. Filter by bus
        res_bus = get_filtered_entry_logs(bus_id="BUS-12", db_path=self.db_path)
        self.assertEqual(len(res_bus), 4)

        # 6. Filter by date (today)
        res_today = get_filtered_entry_logs(start_date=date.today(), end_date=date.today(), db_path=self.db_path)
        self.assertEqual(len(res_today), 3)

    def test_csv_export_structure(self):
        """
        Verifies that exported CSV matches the required columns:
        entry_id, timestamp, student_id, student_name, bus_id, route, similarity, result, reason
        and strictly excludes biometric vectors.
        """
        logs = get_filtered_entry_logs(db_path=self.db_path)
        self.assertGreater(len(logs), 0)

        df = pd.DataFrame(logs)
        export_cols = ["entry_id", "timestamp", "student_id", "student_name", "bus_id", "route", "similarity", "result", "reason"]
        export_df = df[export_cols]

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Verify CSV headers
        reader = csv.reader(io.StringIO(csv_content))
        header = next(reader)
        self.assertEqual(header, export_cols)

        # Ensure no vector/embedding columns are present
        self.assertNotIn("embedding", [col.lower() for col in header])
        self.assertNotIn("vector", [col.lower() for col in header])

        # Verify rows count
        rows = list(reader)
        self.assertEqual(len(rows), len(logs))

    def test_distinct_helpers(self):
        """Verifies that dropdown helper functions return correct distinct values."""
        buses = get_distinct_buses(db_path=self.db_path)
        self.assertIn("BUS-12", buses)
        self.assertIn("BUS-08", buses)

        routes = get_distinct_routes(db_path=self.db_path)
        self.assertIn("R-101", routes)
        self.assertIn("R-102", routes)

        depts = get_all_departments(db_path=self.db_path)
        self.assertIn("Computer Science", depts)
        self.assertIn("Electronics", depts)

        sems = get_all_semesters(db_path=self.db_path)
        self.assertIn("6", sems)
        self.assertIn("4", sems)

    def test_system_status_db_check(self):
        """Verifies SQLite connection diagnostics on valid and non-existent databases."""
        orig_path = sp.DB_PATH
        try:
            sp.DB_PATH = self.db_path
            ok, msg = sp.test_sqlite_db()
            self.assertTrue(ok)
            self.assertIn("Online", msg)

            sp.DB_PATH = Path(self.temp_dir.name) / "non_existent.db"
            bad_ok, bad_msg = sp.test_sqlite_db()
            self.assertFalse(bad_ok)
            self.assertIn("missing", bad_msg.lower())
        finally:
            sp.DB_PATH = orig_path


if __name__ == "__main__":
    unittest.main()
