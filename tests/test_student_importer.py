"""
Unit Tests for Student & Bus Pass Importer
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

from src.data.student_importer import StudentImporter
from src.database.schema import (
    CREATE_STUDENTS_TABLE,
    CREATE_BUS_ROUTES_TABLE,
    CREATE_BUS_PASSES_TABLE,
    CREATE_ENTRY_LOGS_TABLE
)


class TestStudentImporter(unittest.TestCase):
    """Test suite covering manual CSV importing, validation, and database updates."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bus.db"
        self._init_test_database()
        self.importer = StudentImporter(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_test_database(self):
        """Creates empty tables in temporary database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute(CREATE_STUDENTS_TABLE)
        cursor.execute(CREATE_BUS_ROUTES_TABLE)
        cursor.execute(CREATE_BUS_PASSES_TABLE)
        cursor.execute(CREATE_ENTRY_LOGS_TABLE)
        conn.commit()
        conn.close()

    def _write_csv(self, content: str) -> Path:
        """Helper to write CSV string into temporary file."""
        csv_file = Path(self.temp_dir.name) / "test_students.csv"
        csv_file.write_text(content.strip(), encoding="utf-8")
        return csv_file

    def test_valid_student_csv_row(self):
        """Verify valid rows are cleanly imported into students, routes, and passes tables."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,Computer Science,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)

        self.assertEqual(summary.total_rows, 1)
        self.assertEqual(summary.imported_count, 1)
        self.assertEqual(summary.rejected_count, 0)

        # Verify in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT roll_number, name, department, semester FROM students WHERE roll_number = '21CS101'")
        student = cursor.fetchone()
        self.assertIsNotNone(student)
        self.assertEqual(student[0], "21CS101")
        self.assertEqual(student[1], "Aarav Sharma")
        self.assertEqual(student[2], "Computer Science")
        self.assertEqual(student[3], "6")

        cursor.execute("SELECT route_code, bus_number FROM bus_routes WHERE route_code = 'R-101'")
        route = cursor.fetchone()
        self.assertIsNotNone(route)
        self.assertEqual(route[0], "R-101")
        self.assertEqual(route[1], "BUS-12")

        cursor.execute("SELECT status, start_date, expiry_date FROM bus_passes WHERE student_id = 1")
        pass_rec = cursor.fetchone()
        self.assertIsNotNone(pass_rec)
        self.assertEqual(pass_rec[0], "ACTIVE")
        self.assertEqual(pass_rec[1], "2026-07-01")
        self.assertEqual(pass_rec[2], "2026-12-31")
        conn.close()

    def test_missing_student_id_rejected(self):
        """Verify rows missing student_id are rejected."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
,Aarav Sharma,CS,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)
        self.assertEqual(summary.rejected_count, 1)
        self.assertIn("student_id", summary.rejections[0]["reason"])

    def test_missing_name_rejected(self):
        """Verify rows missing student name are rejected."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,,CS,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)
        self.assertEqual(summary.rejected_count, 1)
        self.assertIn("name", summary.rejections[0]["reason"])

    def test_invalid_dates_rejected(self):
        """Verify malformed date strings are caught."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,CS,6,BUS-12,R-101,01-07-2026,2026-12-31,ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)
        self.assertEqual(summary.rejected_count, 1)
        self.assertIn("pass_start date", summary.rejections[0]["reason"])

    def test_pass_start_after_pass_end_rejected(self):
        """Verify pass_start > pass_end is rejected."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,CS,6,BUS-12,R-101,2026-12-31,2026-07-01,ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)
        self.assertEqual(summary.rejected_count, 1)
        self.assertIn("cannot be after pass_end", summary.rejections[0]["reason"])

    def test_invalid_pass_status_rejected(self):
        """Verify invalid pass_status values are rejected."""
        csv_data = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,CS,6,BUS-12,R-101,2026-07-01,2026-12-31,SUPER_ACTIVE
"""
        csv_path = self._write_csv(csv_data)
        summary = self.importer.import_csv(csv_path)
        self.assertEqual(summary.rejected_count, 1)
        self.assertIn("Invalid pass_status", summary.rejections[0]["reason"])

    def test_duplicate_student_id_updates_cleanly(self):
        """Verify importing the same student ID twice updates rather than duplicating."""
        csv_v1 = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,Computer Science,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
"""
        csv_path = self._write_csv(csv_v1)
        summary1 = self.importer.import_csv(csv_path)
        self.assertEqual(summary1.imported_count, 1)
        self.assertEqual(summary1.updated_count, 0)

        # Update Aarav's semester to 7 and route to R-102
        csv_v2 = """student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,Computer Science,7,BUS-08,R-102,2026-07-01,2027-06-30,ACTIVE
"""
        csv_path2 = self._write_csv(csv_v2)
        summary2 = self.importer.import_csv(csv_path2)
        self.assertEqual(summary2.imported_count, 0)
        self.assertEqual(summary2.updated_count, 1)

        # Check single row in students table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students WHERE roll_number = '21CS101'")
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT semester FROM students WHERE roll_number = '21CS101'")
        self.assertEqual(cursor.fetchone()[0], "7")
        conn.close()


if __name__ == "__main__":
    unittest.main()
