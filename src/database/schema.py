"""
Database Schema Definition
Smart Bus Face Recognition and Pass Verification System

Defines the tables required for students, bus routes, bus passes, and entry logs.
"""

from .db_connection import get_db_connection

CREATE_STUDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    photo_path TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_BUS_ROUTES_TABLE = """
CREATE TABLE IF NOT EXISTS bus_routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    route_code TEXT UNIQUE NOT NULL,
    route_name TEXT NOT NULL,
    bus_number TEXT NOT NULL,
    driver_name TEXT
);
"""

CREATE_BUS_PASSES_TABLE = """
CREATE TABLE IF NOT EXISTS bus_passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    route_id INTEGER NOT NULL,
    pass_type TEXT NOT NULL DEFAULT 'Semester',
    start_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (route_id) REFERENCES bus_routes(id) ON DELETE CASCADE
);
"""

CREATE_ENTRY_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS entry_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    route_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verification_method TEXT NOT NULL,
    status TEXT NOT NULL,
    rejection_reason TEXT,
    similarity_score REAL,
    snapshot_path TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL,
    FOREIGN KEY (route_id) REFERENCES bus_routes(id) ON DELETE CASCADE
);
"""

def initialize_tables():
    """Executes all CREATE TABLE statements to set up the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(CREATE_STUDENTS_TABLE)
    cursor.execute(CREATE_BUS_ROUTES_TABLE)
    cursor.execute(CREATE_BUS_PASSES_TABLE)
    cursor.execute(CREATE_ENTRY_LOGS_TABLE)
    
    conn.commit()
    conn.close()
    print("[SUCCESS] All database tables initialized successfully.")

if __name__ == "__main__":
    initialize_tables()
