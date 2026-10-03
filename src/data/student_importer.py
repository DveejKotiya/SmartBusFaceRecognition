"""
Student & Bus Pass CSV Importer
Smart Bus Face Recognition and Pass Verification System

This module reads manually curated student and bus pass records from CSV,
validates each row, and inserts or updates records in the SQLite database.
"""

import sys
import csv
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import sqlite3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_db_connection, DB_PATH
from src.database.schema import initialize_tables

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "students" / "students.csv"
VALID_PASS_STATUSES = {"ACTIVE", "INACTIVE", "EXPIRED"}


@dataclass
class ImportSummary:
    """Summary of the CSV import process."""
    total_rows: int = 0
    imported_count: int = 0
    updated_count: int = 0
    rejected_count: int = 0
    rejections: List[Dict[str, Any]] = field(default_factory=list)


class StudentImporter:
    """
    Imports and synchronizes manual student and bus pass records from CSV into SQLite.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the importer.

        Args:
            db_path: Path to the SQLite database file. Defaults to data/smart_bus.db.
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensures all tables exist and the semester column is present in students."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if students table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
        if not cursor.fetchone():
            conn.close()
            # If database not initialized yet, initialize tables
            initialize_tables()
            conn = self._get_connection()
            cursor = conn.cursor()

        # Ensure semester column exists in students table
        cursor.execute("PRAGMA table_info(students)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "semester" not in columns:
            cursor.execute("ALTER TABLE students ADD COLUMN semester TEXT;")
            conn.commit()

        conn.close()

    def validate_row(self, row: Dict[str, str], row_num: int) -> Tuple[bool, Optional[str]]:
        """
        Validates a single CSV record.

        Args:
            row: Dictionary of CSV values for a row.
            row_num: 1-indexed line number from the CSV file.

        Returns:
            Tuple[bool, Optional[str]]: (True, None) if valid; (False, error_reason) if invalid.
        """
        student_id = row.get("student_id", "").strip()
        name = row.get("name", "").strip()
        route = row.get("route", "").strip()
        bus_id = row.get("bus_id", "").strip()
        pass_start_str = row.get("pass_start", "").strip()
        pass_end_str = row.get("pass_end", "").strip()
        pass_status = row.get("pass_status", "").strip().upper()

        if not student_id:
            return False, f"Row {row_num}: 'student_id' cannot be empty."

        if not name:
            return False, f"Row {row_num}: 'name' cannot be empty for student '{student_id}'."

        if not route:
            return False, f"Row {row_num}: 'route' cannot be empty for student '{student_id}'."

        if not bus_id:
            return False, f"Row {row_num}: 'bus_id' cannot be empty for student '{student_id}'."

        # Validate date formats (YYYY-MM-DD)
        try:
            d_start = datetime.strptime(pass_start_str, "%Y-%m-%d").date()
        except ValueError:
            return False, f"Row {row_num}: Invalid pass_start date '{pass_start_str}' (expected YYYY-MM-DD)."

        try:
            d_end = datetime.strptime(pass_end_str, "%Y-%m-%d").date()
        except ValueError:
            return False, f"Row {row_num}: Invalid pass_end date '{pass_end_str}' (expected YYYY-MM-DD)."

        if d_start > d_end:
            return False, (
                f"Row {row_num}: pass_start ({d_start}) cannot be after pass_end ({d_end}) "
                f"for student '{student_id}'."
            )

        if pass_status not in VALID_PASS_STATUSES:
            return False, (
                f"Row {row_num}: Invalid pass_status '{pass_status}'. "
                f"Must be one of: {', '.join(sorted(VALID_PASS_STATUSES))}."
            )

        return True, None

    def import_csv(self, csv_path: Optional[Union[str, Path]] = None) -> ImportSummary:
        """
        Reads and imports the CSV file into the database.

        Args:
            csv_path: Path to the students CSV file. Defaults to data/students/students.csv.

        Returns:
            ImportSummary: Summary containing total, imported, updated, and rejected counts.
        """
        path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        summary = ImportSummary()

        if not path.exists():
            print(f"[ERROR] CSV file not found at: {path}")
            summary.rejections.append({
                "row_num": 0,
                "data": {},
                "reason": f"File not found: {path}"
            })
            summary.rejected_count = 1
            return summary

        conn = self._get_connection()
        cursor = conn.cursor()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                conn.close()
                return summary

            for idx, raw_row in enumerate(reader, start=2):  # Header is line 1
                # Skip comments or completely empty lines
                first_val = list(raw_row.values())[0] if raw_row.values() else ""
                if str(first_val).strip().startswith("#") or not any(raw_row.values()):
                    continue

                summary.total_rows += 1
                is_valid, error_msg = self.validate_row(raw_row, idx)

                if not is_valid:
                    summary.rejected_count += 1
                    summary.rejections.append({
                        "row_num": idx,
                        "data": raw_row,
                        "reason": error_msg
                    })
                    continue

                # Clean fields
                student_id = raw_row["student_id"].strip().upper()
                name = raw_row["name"].strip()
                department = raw_row.get("department", "").strip() or "General"
                semester = raw_row.get("semester", "").strip() or None
                bus_id = raw_row["bus_id"].strip().upper()
                route_code = raw_row["route"].strip().upper()
                pass_start = raw_row["pass_start"].strip()
                pass_end = raw_row["pass_end"].strip()
                pass_status = raw_row["pass_status"].strip().upper()

                # 1. Resolve or Create Bus Route
                cursor.execute(
                    "SELECT id FROM bus_routes WHERE UPPER(route_code) = ?",
                    (route_code,)
                )
                route_row = cursor.fetchone()
                if route_row:
                    route_id = route_row["id"]
                    # Update bus number if needed
                    cursor.execute(
                        "UPDATE bus_routes SET bus_number = ? WHERE id = ?",
                        (bus_id, route_id)
                    )
                else:
                    cursor.execute("""
                        INSERT INTO bus_routes (route_code, route_name, bus_number)
                        VALUES (?, ?, ?)
                    """, (route_code, f"Route {route_code}", bus_id))
                    route_id = cursor.lastrowid

                # 2. Check if Student exists
                cursor.execute(
                    "SELECT id FROM students WHERE UPPER(roll_number) = ?",
                    (student_id,)
                )
                student_row = cursor.fetchone()

                if student_row:
                    student_pk = student_row["id"]
                    # Update existing student
                    cursor.execute("""
                        UPDATE students 
                        SET name = ?, department = ?, semester = ?, is_active = 1
                        WHERE id = ?
                    """, (name, department, semester, student_pk))
                    summary.updated_count += 1
                else:
                    # Insert new student
                    cursor.execute("""
                        INSERT INTO students (roll_number, name, department, semester, is_active)
                        VALUES (?, ?, ?, ?, 1)
                    """, (student_id, name, department, semester))
                    student_pk = cursor.lastrowid
                    summary.imported_count += 1

                # 3. Synchronize Bus Pass
                cursor.execute(
                    "SELECT id FROM bus_passes WHERE student_id = ? ORDER BY id DESC LIMIT 1",
                    (student_pk,)
                )
                pass_row = cursor.fetchone()

                if pass_row:
                    pass_pk = pass_row["id"]
                    cursor.execute("""
                        UPDATE bus_passes
                        SET route_id = ?, start_date = ?, expiry_date = ?, status = ?
                        WHERE id = ?
                    """, (route_id, pass_start, pass_end, pass_status, pass_pk))
                else:
                    cursor.execute("""
                        INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
                        VALUES (?, ?, 'Semester', ?, ?, ?)
                    """, (student_pk, route_id, pass_start, pass_end, pass_status))

        conn.commit()
        conn.close()
        return summary


def run_cli():
    """Command-line interface for student data import."""
    print("=" * 65)
    print("  SMART BUS MANUAL DATA IMPORTER")
    print("=" * 65)
    print(f"Reading from CSV: {DEFAULT_CSV_PATH}")

    importer = StudentImporter()
    summary = importer.import_csv()

    print("\n--- IMPORT SUMMARY REPORT ---")
    print(f"  Total Rows Evaluated : {summary.total_rows}")
    print(f"  New Students Added   : {summary.imported_count}")
    print(f"  Students Updated     : {summary.updated_count}")
    print(f"  Rows Rejected        : {summary.rejected_count}")

    if summary.rejections:
        print("\n[WARNING] The following rows had errors and were NOT imported:")
        for rej in summary.rejections:
            print(f"  - {rej['reason']}")
    else:
        print("\n[SUCCESS] All records imported and synchronized without error!")

    print("\nNext step: If you added new student photos in data/students/faces/, run:")
    print("  python -m src.vision.build_student_embeddings")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_cli()
