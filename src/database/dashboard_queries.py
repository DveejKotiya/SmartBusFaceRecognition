"""
Dashboard & Reporting Database Queries
Smart Bus Face Recognition and Pass Verification System

Centralizes all reporting, aggregation, and filtering queries for the Streamlit UI.
Guarantees parameterized SQL to prevent SQL injection.
"""

from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sqlite3

from src.database.db_connection import get_db_connection, DB_PATH


def _connect(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Connects to SQLite with row_factory=Row."""
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def get_dashboard_summary(db_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """
    Computes real-time statistics for the dashboard home page.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()

    # Total registered students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    # Active passes
    cursor.execute("""
        SELECT COUNT(*) FROM bus_passes
        WHERE status = 'ACTIVE' AND date(expiry_date) >= date('now', 'localtime')
    """)
    active_passes = cursor.fetchone()[0]

    # Expired passes
    cursor.execute("""
        SELECT COUNT(*) FROM bus_passes
        WHERE status = 'EXPIRED' OR date(expiry_date) < date('now', 'localtime')
    """)
    expired_passes = cursor.fetchone()[0]

    # Today's entries
    cursor.execute("""
        SELECT COUNT(*) FROM entry_logs
        WHERE status = 'APPROVED' AND date(timestamp) = date('now', 'localtime')
    """)
    today_allowed = cursor.fetchone()[0]

    # Today's denied
    cursor.execute("""
        SELECT COUNT(*) FROM entry_logs
        WHERE status = 'REJECTED' AND date(timestamp) = date('now', 'localtime')
    """)
    today_denied = cursor.fetchone()[0]

    # Today's unknown person attempts
    cursor.execute("""
        SELECT COUNT(*) FROM entry_logs
        WHERE rejection_reason = 'UNKNOWN_PERSON' AND date(timestamp) = date('now', 'localtime')
    """)
    today_unknown = cursor.fetchone()[0]

    conn.close()

    return {
        "total_students": total_students,
        "active_passes": active_passes,
        "expired_passes": expired_passes,
        "today_allowed": today_allowed,
        "today_denied": today_denied,
        "today_unknown": today_unknown
    }


def get_recent_activity(limit: int = 10, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the latest boarding attempts across all buses.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT el.id, el.timestamp, el.status AS result, el.rejection_reason AS reason,
               el.similarity_score AS similarity,
               COALESCE(s.name, 'UNKNOWN PERSON') AS student_name,
               COALESCE(s.roll_number, 'N/A') AS student_id,
               br.bus_number AS bus_id, br.route_code AS route
        FROM entry_logs el
        LEFT JOIN students s ON el.student_id = s.id
        JOIN bus_routes br ON el.route_id = br.id
        ORDER BY el.id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_students_directory(
    search: Optional[str] = None,
    department: Optional[str] = None,
    semester: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Returns student profiles with latest bus pass and route details.
    Supports multi-attribute search and filtering.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT s.id, s.roll_number AS student_id, s.name, s.department,
               COALESCE(s.semester, 'N/A') AS semester,
               COALESCE(br.bus_number, 'None') AS bus_id,
               COALESCE(br.route_code, 'None') AS route,
               COALESCE(bp.status, 'NO_PASS') AS pass_status,
               bp.start_date, bp.expiry_date
        FROM students s
        LEFT JOIN bus_passes bp ON bp.student_id = s.id AND bp.id = (
            SELECT id FROM bus_passes WHERE student_id = s.id ORDER BY id DESC LIMIT 1
        )
        LEFT JOIN bus_routes br ON bp.route_id = br.id
        WHERE 1=1
    """
    params: List[Any] = []

    if search:
        search_pattern = f"%{search.strip().upper()}%"
        query += " AND (UPPER(s.roll_number) LIKE ? OR UPPER(s.name) LIKE ?)"
        params.extend([search_pattern, search_pattern])

    if department and department != "All":
        query += " AND s.department = ?"
        params.append(department)

    if semester and semester != "All":
        query += " AND s.semester = ?"
        params.append(semester)

    query += " ORDER BY s.roll_number ASC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_departments(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Lists distinct departments in the student table."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT department FROM students WHERE department IS NOT NULL AND department != '' ORDER BY department")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_all_semesters(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Lists distinct semesters in the student table."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(students)")
    cols = [r["name"] for r in cursor.fetchall()]
    if "semester" not in cols:
        conn.close()
        return []

    cursor.execute("SELECT DISTINCT semester FROM students WHERE semester IS NOT NULL AND semester != '' ORDER BY semester")
    rows = cursor.fetchall()
    conn.close()
    return [str(r[0]) for r in rows]


def get_passes_directory(
    status_filter: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Returns all bus passes with student and route info.
    Computes is_expired flag dynamically based on current date.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT bp.id AS pass_id, s.roll_number AS student_id, s.name AS student_name,
               br.bus_number AS bus_id, br.route_code AS route,
               bp.start_date, bp.expiry_date, bp.status,
               CASE 
                   WHEN date(bp.expiry_date) < date('now', 'localtime') THEN 1 
                   ELSE 0 
               END AS is_expired
        FROM bus_passes bp
        JOIN students s ON bp.student_id = s.id
        JOIN bus_routes br ON bp.route_id = br.id
        WHERE 1=1
    """
    params: List[Any] = []

    if status_filter and status_filter != "ALL":
        if status_filter == "EXPIRED":
            query += " AND (bp.status = 'EXPIRED' OR date(bp.expiry_date) < date('now', 'localtime'))"
        elif status_filter == "ACTIVE":
            query += " AND bp.status = 'ACTIVE' AND date(bp.expiry_date) >= date('now', 'localtime')"
        else:
            query += " AND bp.status = ?"
            params.append(status_filter)

    query += " ORDER BY bp.id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_filtered_entry_logs(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    student_query: Optional[str] = None,
    bus_id: Optional[str] = None,
    route: Optional[str] = None,
    result: Optional[str] = None,
    limit: int = 150,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Searches and filters entry logs with pagination support.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT el.id AS entry_id, el.timestamp,
               COALESCE(s.roll_number, 'N/A') AS student_id,
               COALESCE(s.name, 'UNKNOWN PERSON') AS student_name,
               br.bus_number AS bus_id, br.route_code AS route,
               COALESCE(el.similarity_score, 0.0) AS similarity,
               el.status AS result,
               COALESCE(el.rejection_reason, 'PASS_VALID') AS reason
        FROM entry_logs el
        LEFT JOIN students s ON el.student_id = s.id
        JOIN bus_routes br ON el.route_id = br.id
        WHERE 1=1
    """
    params: List[Any] = []

    if start_date:
        query += " AND date(el.timestamp) >= date(?)"
        params.append(start_date.isoformat())

    if end_date:
        query += " AND date(el.timestamp) <= date(?)"
        params.append(end_date.isoformat())

    if student_query:
        pattern = f"%{student_query.strip().upper()}%"
        query += " AND (UPPER(s.roll_number) LIKE ? OR UPPER(s.name) LIKE ?)"
        params.extend([pattern, pattern])

    if bus_id and bus_id != "All":
        query += " AND UPPER(br.bus_number) = ?"
        params.append(bus_id.strip().upper())

    if route and route != "All":
        query += " AND UPPER(br.route_code) = ?"
        params.append(route.strip().upper())

    if result and result != "ALL":
        if result == "UNKNOWN":
            query += " AND el.rejection_reason = 'UNKNOWN_PERSON'"
        else:
            query += " AND el.status = ?"
            params.append(result)

    query += " ORDER BY el.id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_distinct_buses(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Returns list of all active bus numbers."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT bus_number FROM bus_routes WHERE bus_number IS NOT NULL ORDER BY bus_number")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_distinct_routes(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Returns list of all active route codes."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT route_code FROM bus_routes WHERE route_code IS NOT NULL ORDER BY route_code")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]
