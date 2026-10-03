"""
Database Initialization & Seeding Script
Smart Bus Face Recognition and Pass Verification System

Initializes SQLite tables and seeds sample routes, students, and passes for testing.
"""

from datetime import datetime, timedelta
from .schema import initialize_tables
from .crud import (
    create_route, get_all_routes,
    create_student, get_all_students,
    create_bus_pass, get_student_active_pass,
    log_entry, get_recent_logs
)

def seed_sample_data():
    """Seeds sample routes, students, and passes for testing."""
    print("\n--- Seeding Initial Data ---")
    
    # 1. Seed Bus Routes
    routes = get_all_routes()
    if not routes:
        r1 = create_route("R-101", "Downtown to Main Campus", "BUS-12", "Ramesh Kumar")
        r2 = create_route("R-102", "Railway Station to Tech Park", "BUS-08", "Suresh Singh")
        print(f"  [+] Created Route R-101 (ID: {r1}) and Route R-102 (ID: {r2})")
    else:
        print(f"  [i] {len(routes)} routes already exist.")

    # Fetch route IDs
    routes = get_all_routes()
    r101_id = next(r["id"] for r in routes if r["route_code"] == "R-101")
    r102_id = next(r["id"] for r in routes if r["route_code"] == "R-102")

    # 2. Seed Test Students
    students = get_all_students()
    if not students:
        today = datetime.now().date()
        valid_expiry = (today + timedelta(days=180)).strftime("%Y-%m-%d")
        expired_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")

        # Student 1: Valid Pass on R-101
        s1 = create_student("21CS101", "Aarav Sharma", "Computer Science", "aarav@college.edu", "9876543210")
        p1 = create_bus_pass(s1, r101_id, start_date, valid_expiry, "Semester", "ACTIVE")
        print(f"  [+] Seeded Student 'Aarav Sharma' (21CS101) with VALID pass until {valid_expiry}")

        # Student 2: Expired Pass on R-101
        s2 = create_student("21EC202", "Priya Patel", "Electronics", "priya@college.edu", "9876543211")
        p2 = create_bus_pass(s2, r101_id, start_date, expired_date, "Semester", "EXPIRED")
        print(f"  [+] Seeded Student 'Priya Patel' (21EC202) with EXPIRED pass (expired {expired_date})")

        # Student 3: Valid Pass on R-102 (Wrong Route if boarding R-101)
        s3 = create_student("21ME303", "Rohan Gupta", "Mechanical", "rohan@college.edu", "9876543212")
        p3 = create_bus_pass(s3, r102_id, start_date, valid_expiry, "Semester", "ACTIVE")
        print(f"  [+] Seeded Student 'Rohan Gupta' (21ME303) with pass on Route R-102")
    else:
        print(f"  [i] {len(students)} students already exist.")

    # 3. Test Entry Log
    log_id = log_entry(
        route_id=r101_id,
        status="APPROVED",
        verification_method="MANUAL_FALLBACK",
        student_id=1,
        similarity_score=1.0
    )
    print(f"  [+] Created test entry log (ID: {log_id})")

    # 4. Verification Check
    recent_logs = get_recent_logs(5)
    print(f"\n[SUCCESS] Database seeded and verified. Recent logs count: {len(recent_logs)}")

def main():
    print("=" * 60)
    print("  INITIALIZING SMART BUS SQLITE DATABASE")
    print("=" * 60)
    initialize_tables()
    seed_sample_data()
    print("=" * 60)

if __name__ == "__main__":
    main()
