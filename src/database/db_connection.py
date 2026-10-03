"""
Database Connection Manager
Smart Bus Face Recognition and Pass Verification System
"""

import sqlite3
from pathlib import Path

# Project paths: resolves to project root/data/smart_bus.db
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "smart_bus.db"

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    - Automatically creates the data/ folder if it doesn't exist.
    - Enables foreign key enforcement.
    - Sets row_factory = sqlite3.Row so query results behave like Python dictionaries.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    # Enable foreign keys in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    # Allows accessing columns by name: row['roll_number'] instead of row[1]
    conn.row_factory = sqlite3.Row
    return conn
