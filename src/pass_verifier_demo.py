"""
Pass Verifier Demo Script
Smart Bus Face Recognition and Pass Verification System

This demo tests the PassVerifier business rules engine across 5 canonical scenarios
using deterministic test data (without needing webcam hardware):

CASE 1: Recognized student + valid pass + correct route -> ALLOWED
CASE 2: Recognized student + expired pass -> DENIED (PASS_EXPIRED)
CASE 3: Recognized student + wrong route -> DENIED (ROUTE_MISMATCH)
CASE 4: Unknown person -> DENIED (UNKNOWN_PERSON)
CASE 5: Recognized student attempting to board again within 5 minutes -> DENIED (DUPLICATE_COOLDOWN)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pass_verifier import (
    PassVerifier,
    RecognitionResult,
    VerificationResult,
    ReasonCode
)
from src.database.init_db import main as init_database


def format_result(case_num: int, title: str, res: VerificationResult) -> None:
    """Pretty prints a verification result."""
    status_icon = "🟢 ALLOWED" if res.allowed else "🔴 DENIED"
    print("\n" + "-" * 60)
    print(f"  CASE {case_num}: {title}")
    print("-" * 60)
    print(f"  Status        : {status_icon}")
    print(f"  Student ID    : {res.student_id}")
    print(f"  Student Name  : {res.student_name}")
    print(f"  Similarity    : {res.similarity:.2f}")
    print(f"  Bus / Route   : {res.bus_id} ({res.route})")
    print(f"  Reason Code   : {res.reason_code.value}")
    print(f"  Message       : {res.reason_message}")
    print(f"  Timestamp     : {res.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duplicate?    : {res.duplicate}")


def run_demo():
    print("=" * 65)
    print("  SMART BUS FACE RECOGNITION - PASS VERIFIER DEMO (STAGE 4)")
    print("=" * 65)

    # 1. Initialize & Seed Database with test records
    print("\n[Step 1] Ensuring SQLite database is seeded...")
    init_database()

    # 2. Instantiate PassVerifier
    verifier = PassVerifier(recognition_threshold=0.60, cooldown_seconds=300)
    t0 = datetime.now(timezone.utc)

    # -------------------------------------------------------------
    # CASE 1: Recognized student + valid pass + correct route
    # -------------------------------------------------------------
    # Aarav Sharma (21CS101) holds an active pass on Route R-101 (BUS-12)
    rec1 = RecognitionResult(
        recognized=True,
        student_id="21CS101",
        student_name="Aarav Sharma",
        similarity=0.92
    )
    res1 = verifier.verify(rec1, bus_id="BUS-12", route="R-101", current_timestamp=t0)
    format_result(1, "Recognized Student + Valid Pass + Correct Route", res1)
    # Log the successful entry to test duplicate cooldown in Case 5
    verifier.log_verification(res1)

    # -------------------------------------------------------------
    # CASE 2: Recognized student + expired pass
    # -------------------------------------------------------------
    # Priya Patel (21EC202) has an expired pass on Route R-101
    rec2 = RecognitionResult(
        recognized=True,
        student_id="21EC202",
        student_name="Priya Patel",
        similarity=0.89
    )
    res2 = verifier.verify(rec2, bus_id="BUS-12", route="R-101", current_timestamp=t0)
    format_result(2, "Recognized Student + Expired Pass", res2)

    # -------------------------------------------------------------
    # CASE 3: Recognized student + wrong route
    # -------------------------------------------------------------
    # Rohan Gupta (21ME303) holds pass on Route R-102 (BUS-08), but tries to board BUS-12 (R-101)
    rec3 = RecognitionResult(
        recognized=True,
        student_id="21ME303",
        student_name="Rohan Gupta",
        similarity=0.94
    )
    res3 = verifier.verify(rec3, bus_id="BUS-12", route="R-101", current_timestamp=t0)
    format_result(3, "Recognized Student + Wrong Route", res3)

    # -------------------------------------------------------------
    # CASE 4: Unknown person
    # -------------------------------------------------------------
    rec4 = RecognitionResult(
        recognized=False,
        student_id=None,
        student_name=None,
        similarity=0.25
    )
    res4 = verifier.verify(rec4, bus_id="BUS-12", route="R-101", current_timestamp=t0)
    format_result(4, "Unknown Person (Face not recognized)", res4)

    # -------------------------------------------------------------
    # CASE 5: Recognized student attempting to board again within 5 minutes
    # -------------------------------------------------------------
    # Aarav Sharma scans again 2 minutes (120 seconds) after Case 1
    t_duplicate = t0 + timedelta(seconds=120)
    res5 = verifier.verify(rec1, bus_id="BUS-12", route="R-101", current_timestamp=t_duplicate)
    format_result(5, "Duplicate Boarding Attempt within 5-Minute Cooldown", res5)

    print("\n" + "=" * 65)
    print("  PASS VERIFIER DEMO COMPLETE: ALL 5 RULES DEMONSTRATED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
