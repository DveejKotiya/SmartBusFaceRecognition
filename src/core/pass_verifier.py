"""
Pass Verifier & Business Rules Engine
Smart Bus Face Recognition and Pass Verification System

This module bridges face recognition with database business logic to decide whether
a passenger is allowed to board the bus.

Responsibilities:
- Evaluates recognition confidence against a configurable threshold.
- Validates student existence and active status in the database.
- Checks bus pass existence, active state, and date validity (inclusive expiration).
- Validates route matching between student pass and the current bus.
- Enforces a 5-minute database-backed anti-duplicate boarding cooldown window.
- Returns structured VerificationResult with deterministic reason codes.
"""

from dataclasses import dataclass
from datetime import datetime, date, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Any
import sqlite3

from src.database.db_connection import get_db_connection, DB_PATH


# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

# Default recognition similarity threshold (cosine similarity in [-1.0, 1.0]).
# Cosine similarities >= 0.60 indicate high facial confidence with FaceNet VGGFace2.
# This threshold is configurable and should be tuned experimentally.
DEFAULT_RECOGNITION_THRESHOLD: float = 0.60

# Anti-duplicate boarding cooldown period in seconds (5 minutes)
DEFAULT_COOLDOWN_SECONDS: int = 300


class ReasonCode(str, Enum):
    """Standardized deterministic reason codes for boarding verification."""
    PASS_VALID = "PASS_VALID"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STUDENT_NOT_FOUND = "STUDENT_NOT_FOUND"
    STUDENT_INACTIVE = "STUDENT_INACTIVE"
    NO_PASS = "NO_PASS"
    PASS_INACTIVE = "PASS_INACTIVE"
    PASS_NOT_STARTED = "PASS_NOT_STARTED"
    PASS_EXPIRED = "PASS_EXPIRED"
    ROUTE_MISMATCH = "ROUTE_MISMATCH"
    DUPLICATE_COOLDOWN = "DUPLICATE_COOLDOWN"
    DATABASE_ERROR = "DATABASE_ERROR"
    LIVENESS_FAILED = "LIVENESS_FAILED"
    LIVENESS_INCONCLUSIVE = "LIVENESS_INCONCLUSIVE"


# =====================================================================
# DATA CONTAINERS
# =====================================================================

@dataclass
class RecognitionResult:
    """
    Structured input from the face recognition module.

    Attributes:
        recognized: True if face matched an identity in the store, False otherwise.
        student_id: The primary student identifier (Roll Number or database ID).
        student_name: Student full name (informational only; ID is authoritative).
        similarity: Cosine similarity score between -1.0 and 1.0.
    """
    recognized: bool
    student_id: Optional[Union[int, str]] = None
    student_name: Optional[str] = None
    similarity: float = 0.0


@dataclass
class VerificationResult:
    """
    Structured outcome of the pass verification rules engine.

    Attributes:
        allowed: True if all checks pass and boarding is authorized; False otherwise.
        student_id: The verified student ID (or None if unknown).
        student_name: The student's official name from the database (or None).
        similarity: The facial similarity score recorded during recognition.
        bus_id: Identifier of the bus (e.g. 'BUS-12').
        route: Identifier of the route (e.g. 'R-101').
        reason_code: Machine-readable reason code from ReasonCode.
        reason_message: Human-friendly explanation for screen display or audio cue.
        timestamp: Time of the verification decision.
        duplicate: True if rejected specifically due to the 5-minute cooldown.
    """
    allowed: bool
    student_id: Optional[Union[int, str]]
    student_name: Optional[str]
    similarity: float
    bus_id: str
    route: str
    reason_code: ReasonCode
    reason_message: str
    timestamp: datetime
    duplicate: bool = False


# =====================================================================
# PASS VERIFIER ENGINE
# =====================================================================

class PassVerifier:
    """
    Evaluates passenger boarding eligibility by applying sequential business rules.

    Architecture separation:
    - Face Recognition: Extracts features & computes similarity.
    - PassVerifier: Enforces database rules (identity, pass validity, route, cooldown).
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    ) -> None:
        """
        Initializes the PassVerifier.

        Args:
            db_path: Optional custom path to SQLite database (defaults to data/smart_bus.db).
            recognition_threshold: Minimum cosine similarity required to accept face match.
            cooldown_seconds: Cooldown window to prevent duplicate boarding logs (default 300s).
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.recognition_threshold = recognition_threshold
        self.cooldown_seconds = cooldown_seconds

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a safe SQLite connection to the configured database file."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _parse_db_date(self, date_val: Union[str, date, datetime]) -> date:
        """Parses a date string ('YYYY-MM-DD') or date object into a datetime.date."""
        if isinstance(date_val, datetime):
            return date_val.date()
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            # Strip time part if present
            clean_str = date_val.strip().split(" ")[0].split("T")[0]
            return datetime.strptime(clean_str, "%Y-%m-%d").date()
        raise ValueError(f"Cannot parse date value: {date_val}")

    def _parse_db_timestamp(self, ts_val: Union[str, datetime]) -> datetime:
        """Parses an SQLite timestamp into a timezone-aware UTC datetime."""
        if isinstance(ts_val, datetime):
            if ts_val.tzinfo is None:
                return ts_val.replace(tzinfo=timezone.utc)
            return ts_val.astimezone(timezone.utc)

        ts_str = str(ts_val).strip()
        # Handle formats: "YYYY-MM-DD HH:MM:SS" or ISO
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Fallback using fromisoformat
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _find_route_in_db(self, conn: sqlite3.Connection, bus_id: str, route: str) -> Optional[sqlite3.Row]:
        """
        Locates the route record in the database by route_code, bus_number, or ID.
        """
        cursor = conn.cursor()
        # 1. Search by exact route_code (e.g. 'R-101')
        cursor.execute("SELECT * FROM bus_routes WHERE UPPER(route_code) = ?", (route.strip().upper(),))
        row = cursor.fetchone()
        if row:
            return row

        # 2. Search by bus_number (e.g. 'BUS-12')
        cursor.execute("SELECT * FROM bus_routes WHERE UPPER(bus_number) = ?", (bus_id.strip().upper(),))
        row = cursor.fetchone()
        if row:
            return row

        # 3. Search by route integer ID if numeric
        if route.strip().isdigit():
            cursor.execute("SELECT * FROM bus_routes WHERE id = ?", (int(route.strip()),))
            row = cursor.fetchone()
            if row:
                return row

        return None

    def _find_student_in_db(self, conn: sqlite3.Connection, student_id: Union[int, str]) -> Optional[sqlite3.Row]:
        """
        Retrieves a student record from the database by roll_number or primary key ID.
        """
        cursor = conn.cursor()
        # 1. Search by roll_number (case-insensitive string like '21CS101')
        cursor.execute("SELECT * FROM students WHERE UPPER(roll_number) = ?", (str(student_id).strip().upper(),))
        row = cursor.fetchone()
        if row:
            return row

        # 2. Search by primary key id if numeric
        if str(student_id).strip().isdigit():
            cursor.execute("SELECT * FROM students WHERE id = ?", (int(student_id),))
            row = cursor.fetchone()
            if row:
                return row

        return None

    def _get_active_pass_in_db(self, conn: sqlite3.Connection, student_pk_id: int) -> Optional[sqlite3.Row]:
        """
        Retrieves the latest bus pass for a student, joining the route details.
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bp.id AS pass_id, bp.student_id, bp.pass_type, bp.start_date, 
                   bp.expiry_date, bp.status AS pass_status,
                   br.id AS route_id, br.route_code, br.route_name, br.bus_number
            FROM bus_passes bp
            JOIN bus_routes br ON bp.route_id = br.id
            WHERE bp.student_id = ?
            ORDER BY bp.id DESC
            LIMIT 1
        """, (student_pk_id,))
        return cursor.fetchone()

    def _check_duplicate_cooldown(
        self,
        conn: sqlite3.Connection,
        student_pk_id: int,
        route_pk_id: int,
        current_dt: datetime
    ) -> Optional[float]:
        """
        Checks if the student had a previous APPROVED boarding on this route within cooldown window.

        Returns:
            Optional[float]: Elapsed seconds since last approved entry if within cooldown; None if eligible.
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp FROM entry_logs
            WHERE student_id = ? AND route_id = ? AND status = 'APPROVED'
            ORDER BY id DESC
            LIMIT 1
        """, (student_pk_id, route_pk_id))
        row = cursor.fetchone()

        if not row:
            return None

        last_time = self._parse_db_timestamp(row["timestamp"])
        elapsed = (current_dt - last_time).total_seconds()

        # Strict boundary: strictly less than cooldown window triggers rejection
        if elapsed < self.cooldown_seconds:
            return elapsed

        return None

    def verify(
        self,
        recognition: RecognitionResult,
        bus_id: str,
        route: str,
        current_timestamp: Optional[datetime] = None
    ) -> VerificationResult:
        """
        Executes sequential business validation rules to determine boarding eligibility.

        Rule Evaluation Order:
        1. Recognition Check: Face recognized?
        2. Confidence Check: Similarity >= recognition_threshold?
        3. Student Existence: Student exists in database?
        4. Student Status: Student account active?
        5. Pass Existence: Student has a bus pass on record?
        6. Pass Status: Pass status == 'ACTIVE'?
        7. Date Validity: Start date <= current date <= Expiry date (inclusive)?
        8. Route Matching: Pass route matches the bus route?
        9. Anti-Duplicate: Has the student already boarded this bus within cooldown window?

        Args:
            recognition: Structured recognition output containing match and similarity.
            bus_id: Current bus identifier (e.g. 'BUS-12').
            route: Current route identifier (e.g. 'R-101').
            current_timestamp: Optional datetime of check (defaults to UTC now).

        Returns:
            VerificationResult: Complete structured decision with deterministic reason codes.
        """
        # Ensure timezone-aware evaluation timestamp
        if current_timestamp is None:
            now_dt = datetime.now(timezone.utc)
        elif current_timestamp.tzinfo is None:
            now_dt = current_timestamp.replace(tzinfo=timezone.utc)
        else:
            now_dt = current_timestamp.astimezone(timezone.utc)

        eval_date = now_dt.date()

        # -------------------------------------------------------------
        # RULE 1: Face Recognition Result Check
        # -------------------------------------------------------------
        if not recognition.recognized or recognition.student_id is None:
            return VerificationResult(
                allowed=False,
                student_id=None,
                student_name=None,
                similarity=recognition.similarity,
                bus_id=bus_id,
                route=route,
                reason_code=ReasonCode.UNKNOWN_PERSON,
                reason_message="Face not recognized in registered database.",
                timestamp=now_dt
            )

        # -------------------------------------------------------------
        # RULE 2: Recognition Confidence Threshold Check
        # -------------------------------------------------------------
        if recognition.similarity < self.recognition_threshold:
            return VerificationResult(
                allowed=False,
                student_id=recognition.student_id,
                student_name=recognition.student_name,
                similarity=recognition.similarity,
                bus_id=bus_id,
                route=route,
                reason_code=ReasonCode.LOW_CONFIDENCE,
                reason_message=(
                    f"Match confidence {recognition.similarity:.2f} is below "
                    f"the required threshold of {self.recognition_threshold:.2f}."
                ),
                timestamp=now_dt
            )

        # Database checks
        try:
            conn = self._get_connection()
        except Exception as e:
            return VerificationResult(
                allowed=False,
                student_id=recognition.student_id,
                student_name=recognition.student_name,
                similarity=recognition.similarity,
                bus_id=bus_id,
                route=route,
                reason_code=ReasonCode.DATABASE_ERROR,
                reason_message=f"Database connection error: {e}",
                timestamp=now_dt
            )

        try:
            # ---------------------------------------------------------
            # RULE 3: Student Existence in Database
            # ---------------------------------------------------------
            student_row = self._find_student_in_db(conn, recognition.student_id)
            if not student_row:
                return VerificationResult(
                    allowed=False,
                    student_id=recognition.student_id,
                    student_name=recognition.student_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.STUDENT_NOT_FOUND,
                    reason_message=f"Student ID '{recognition.student_id}' does not exist in the database.",
                    timestamp=now_dt
                )

            student_pk_id = student_row["id"]
            authoritative_name = student_row["name"]
            official_roll = student_row["roll_number"]

            # ---------------------------------------------------------
            # RULE 4: Student Active Status Check
            # ---------------------------------------------------------
            if student_row["is_active"] != 1:
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.STUDENT_INACTIVE,
                    reason_message="Student profile is deactivated or suspended.",
                    timestamp=now_dt
                )

            # Resolve the current bus route in database
            current_route_row = self._find_route_in_db(conn, bus_id, route)
            route_pk_id = current_route_row["id"] if current_route_row else None

            # ---------------------------------------------------------
            # RULE 5: Bus Pass Existence Check
            # ---------------------------------------------------------
            pass_row = self._get_active_pass_in_db(conn, student_pk_id)
            if not pass_row:
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.NO_PASS,
                    reason_message="No bus pass found for this student.",
                    timestamp=now_dt
                )

            # ---------------------------------------------------------
            # RULE 6: Pass Status Active Check
            # ---------------------------------------------------------
            pass_status = str(pass_row["pass_status"]).strip().upper()
            if pass_status != "ACTIVE":
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.PASS_INACTIVE,
                    reason_message=f"Bus pass is not active (Status: {pass_status}).",
                    timestamp=now_dt
                )

            # ---------------------------------------------------------
            # RULE 7: Date Validity (Inclusive Expiration Convention)
            # ---------------------------------------------------------
            start_date = self._parse_db_date(pass_row["start_date"])
            expiry_date = self._parse_db_date(pass_row["expiry_date"])

            if eval_date < start_date:
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.PASS_NOT_STARTED,
                    reason_message=f"Pass is not valid yet. Begins on {start_date}.",
                    timestamp=now_dt
                )

            # Inclusive convention: Pass is valid through the entire expiry_date
            if eval_date > expiry_date:
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.PASS_EXPIRED,
                    reason_message=f"Pass expired on {expiry_date}.",
                    timestamp=now_dt
                )

            # ---------------------------------------------------------
            # RULE 8: Route Matching Check
            # ---------------------------------------------------------
            pass_route_code = str(pass_row["route_code"]).strip().upper()
            pass_bus_number = str(pass_row["bus_number"]).strip().upper()
            req_route_norm = route.strip().upper()
            req_bus_norm = bus_id.strip().upper()

            # Pass matches if route code matches OR bus number matches
            route_matches = (
                pass_route_code == req_route_norm or
                pass_bus_number == req_bus_norm or
                (route_pk_id is not None and pass_row["route_id"] == route_pk_id)
            )

            if not route_matches:
                return VerificationResult(
                    allowed=False,
                    student_id=official_roll,
                    student_name=authoritative_name,
                    similarity=recognition.similarity,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.ROUTE_MISMATCH,
                    reason_message=(
                        f"Route mismatch. Pass is valid for route '{pass_route_code}' ({pass_bus_number}), "
                        f"not current bus '{bus_id}' ({route})."
                    ),
                    timestamp=now_dt
                )

            # ---------------------------------------------------------
            # RULE 9: Five-Minute Anti-Duplicate Boarding Cooldown
            # ---------------------------------------------------------
            if route_pk_id is not None:
                elapsed = self._check_duplicate_cooldown(conn, student_pk_id, route_pk_id, now_dt)
                if elapsed is not None:
                    remaining = int(self.cooldown_seconds - elapsed)
                    return VerificationResult(
                        allowed=False,
                        student_id=official_roll,
                        student_name=authoritative_name,
                        similarity=recognition.similarity,
                        bus_id=bus_id,
                        route=route,
                        reason_code=ReasonCode.DUPLICATE_COOLDOWN,
                        reason_message=(
                            f"Duplicate boarding scan. Already boarded {int(elapsed)}s ago. "
                            f"Wait {remaining}s."
                        ),
                        timestamp=now_dt,
                        duplicate=True
                    )

            # ---------------------------------------------------------
            # SUCCESS: All Verification Rules Passed!
            # ---------------------------------------------------------
            return VerificationResult(
                allowed=True,
                student_id=official_roll,
                student_name=authoritative_name,
                similarity=recognition.similarity,
                bus_id=bus_id,
                route=route,
                reason_code=ReasonCode.PASS_VALID,
                reason_message=f"Welcome aboard, {authoritative_name}!",
                timestamp=now_dt,
                duplicate=False
            )

        except Exception as e:
            return VerificationResult(
                allowed=False,
                student_id=recognition.student_id,
                student_name=recognition.student_name,
                similarity=recognition.similarity,
                bus_id=bus_id,
                route=route,
                reason_code=ReasonCode.DATABASE_ERROR,
                reason_message=f"Database query failure: {e}",
                timestamp=now_dt
            )
        finally:
            conn.close()

    def log_verification(
        self,
        result: VerificationResult,
        verification_method: str = "FACE_RECOGNITION",
        snapshot_path: Optional[str] = None
    ) -> int:
        """
        Persists the verification result into the SQLite entry_logs table.

        Args:
            result: The VerificationResult to log.
            verification_method: 'FACE_RECOGNITION' or 'MANUAL_FALLBACK'.
            snapshot_path: Optional path to captured face snapshot.

        Returns:
            int: The primary key ID of the created log entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Resolve student internal PK ID from roll_number if available
        student_pk_id = None
        if result.student_id:
            cursor.execute("SELECT id FROM students WHERE UPPER(roll_number) = ?", (str(result.student_id).strip().upper(),))
            s_row = cursor.fetchone()
            if s_row:
                student_pk_id = s_row["id"]

        # Resolve route internal PK ID
        route_row = self._find_route_in_db(conn, result.bus_id, result.route)
        route_pk_id = route_row["id"] if route_row else 1

        status_str = "APPROVED" if result.allowed else "REJECTED"
        rejection_reason = None if result.allowed else str(result.reason_code)

        # Format timestamp in SQLite standard UTC format
        ts_str = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO entry_logs (student_id, route_id, timestamp, verification_method,
                                    status, rejection_reason, similarity_score, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_pk_id, route_pk_id, ts_str, verification_method,
              status_str, rejection_reason, result.similarity, snapshot_path))

        conn.commit()
        log_id = cursor.lastrowid
        conn.close()
        return log_id
