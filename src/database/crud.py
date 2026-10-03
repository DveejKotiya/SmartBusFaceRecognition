"""
Database CRUD Operations
Smart Bus Face Recognition and Pass Verification System

Provides helper functions to Create, Read, Update, and Delete records in SQLite.
All queries use parameter binding (?) to prevent SQL injection.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from .db_connection import get_db_connection

# ==========================================
# 1. BUS ROUTES OPERATIONS
# ==========================================

def create_route(route_code: str, route_name: str, bus_number: str, driver_name: Optional[str] = None) -> int:
    """Inserts a new bus route. Returns the inserted route ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bus_routes (route_code, route_name, bus_number, driver_name)
        VALUES (?, ?, ?, ?)
    """, (route_code.strip(), route_name.strip(), bus_number.strip(), driver_name))
    conn.commit()
    route_id = cursor.lastrowid
    conn.close()
    return route_id

def get_all_routes() -> List[Dict[str, Any]]:
    """Returns a list of all bus routes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bus_routes ORDER BY route_code ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_route_by_id(route_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a single bus route by its database ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bus_routes WHERE id = ?", (route_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_route_by_code(route_code: str) -> Optional[Dict[str, Any]]:
    """Fetches a single bus route by its route code (e.g. 'R-101')."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bus_routes WHERE route_code = ?", (route_code.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# ==========================================
# 2. STUDENTS OPERATIONS
# ==========================================

def create_student(roll_number: str, name: str, department: str, 
                   email: Optional[str] = None, phone: Optional[str] = None,
                   photo_path: Optional[str] = None) -> int:
    """Inserts a new student. Returns the inserted student ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (roll_number, name, department, email, phone, photo_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (roll_number.strip().upper(), name.strip(), department.strip(), email, phone, photo_path))
    conn.commit()
    student_id = cursor.lastrowid
    conn.close()
    return student_id

def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a student by primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_student_by_roll(roll_number: str) -> Optional[Dict[str, Any]]:
    """Fetches a student by college roll number (case-insensitive)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE UPPER(roll_number) = ?", (roll_number.strip().upper(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_students() -> List[Dict[str, Any]]:
    """Returns all registered students."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY roll_number ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# ==========================================
# 3. BUS PASS OPERATIONS
# ==========================================

def create_bus_pass(student_id: int, route_id: int, start_date: str, 
                    expiry_date: str, pass_type: str = "Semester", 
                    status: str = "ACTIVE") -> int:
    """
    Creates a new bus pass for a student.
    start_date and expiry_date format: 'YYYY-MM-DD'
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bus_passes (student_id, route_id, pass_type, start_date, expiry_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (student_id, route_id, pass_type, start_date, expiry_date, status))
    conn.commit()
    pass_id = cursor.lastrowid
    conn.close()
    return pass_id

def get_student_active_pass(student_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves the latest bus pass for a student along with route details.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT bp.id AS pass_id, bp.student_id, bp.pass_type, bp.start_date, bp.expiry_date,
               bp.status AS pass_status, br.id AS route_id, br.route_code, br.route_name,
               br.bus_number
        FROM bus_passes bp
        JOIN bus_routes br ON bp.route_id = br.id
        WHERE bp.student_id = ?
        ORDER BY bp.id DESC
        LIMIT 1
    """, (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_passes() -> List[Dict[str, Any]]:
    """Fetches all issued bus passes with student and route info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT bp.id AS pass_id, s.roll_number, s.name AS student_name, s.department,
               br.route_code, br.route_name, bp.pass_type, bp.start_date, bp.expiry_date, bp.status
        FROM bus_passes bp
        JOIN students s ON bp.student_id = s.id
        JOIN bus_routes br ON bp.route_id = br.id
        ORDER BY bp.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# ==========================================
# 4. ENTRY LOGS OPERATIONS
# ==========================================

def log_entry(route_id: int, status: str, verification_method: str = "FACE_RECOGNITION",
              student_id: Optional[int] = None, rejection_reason: Optional[str] = None,
              similarity_score: Optional[float] = None, snapshot_path: Optional[str] = None) -> int:
    """
    Records an entry attempt at a bus door.
    status: 'APPROVED' or 'REJECTED'
    verification_method: 'FACE_RECOGNITION' or 'MANUAL_FALLBACK'
    rejection_reason: 'EXPIRED_PASS', 'WRONG_ROUTE', 'UNKNOWN_FACE', 'DUPLICATE_SCAN', etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO entry_logs (student_id, route_id, verification_method, status, 
                                rejection_reason, similarity_score, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, route_id, verification_method, status, 
          rejection_reason, similarity_score, snapshot_path))
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id

def get_last_approved_entry(student_id: int, route_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches the most recent APPROVED entry for a student on a specific route.
    Used for cooldown (duplicate entry prevention).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM entry_logs
        WHERE student_id = ? AND route_id = ? AND status = 'APPROVED'
        ORDER BY timestamp DESC
        LIMIT 1
    """, (student_id, route_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetches the latest entry logs with student details and route info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT el.id, el.timestamp, el.verification_method, el.status, 
               el.rejection_reason, el.similarity_score, el.snapshot_path,
               s.roll_number, s.name AS student_name,
               br.route_code, br.route_name, br.bus_number
        FROM entry_logs el
        LEFT JOIN students s ON el.student_id = s.id
        JOIN bus_routes br ON el.route_id = br.id
        ORDER BY el.timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
