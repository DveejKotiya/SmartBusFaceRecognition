# Stage 4 Report: Pass Verification & Business Rules Engine
## AI-Based Face Recognition and Smart Bus Pass Verification System

---

## 1. Executive Summary

Stage 4 connects the face recognition outputs from Stage 3 with the SQLite database from Stage 2. It implements the business rules layer (`src/core/pass_verifier.py`) that decides whether a passenger is authorized to board a specific bus.

In accordance with system design principles:
- **Face Recognition** remains purely an embedding extraction and similarity calculator.
- **Pass Verifier** acts as the authoritative decision maker, enforcing student validity, pass status, date ranges, route authorization, and anti-duplicate boarding cooldown.

---

## 2. Files Created & Modified

| File | Purpose | Status |
| :--- | :--- | :---: |
| `src/core/pass_verifier.py` | PassVerifier class, RecognitionResult, VerificationResult, and ReasonCode enums | **Created** |
| `src/core/__init__.py` | Core package exports | **Modified** |
| `tests/test_pass_verifier.py` | Unit test suite covering 20 test cases, cooldown persistence, and boundaries | **Created** |
| `src/pass_verifier_demo.py` | Standalone demonstration script showing all 5 canonical boarding cases | **Created** |
| `docs/pass_verification.md` | Beginner-friendly architectural and rule explanation guide | **Created** |
| `TODO.md` | Updated roadmap marking Stage 4 complete | **Modified** |

---

## 3. Existing Components Reused

1. **`src/database/schema.py` & `src/database/db_connection.py`**:
   - Reused `students`, `bus_routes`, `bus_passes`, and `entry_logs` tables without requiring any schema migrations.
   - Reused foreign keys and SQLite connection settings.
2. **`src/database/crud.py`**:
   - Reused existing table structures and verified compatibility with `get_last_approved_entry()` and `log_entry()`.
3. **`src/vision/`**:
   - Reused cosine similarity convention (`[-1.0, 1.0]`) and 512-D embedding structures from `FaceRecognizer`.

---

## 4. Business Rules Implemented & Evaluation Hierarchy

When a passenger is scanned, the `PassVerifier` applies 9 sequential rules in strict order:

```
[Camera Scan / Face Recognition Result]
                  │
                  ▼
┌───────────────────────────────────────┐
│ Rule 1: Recognized Face Present?      │ ── NO ──> REJECT: UNKNOWN_PERSON
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 2: Similarity >= Threshold?      │ ── NO ──> REJECT: LOW_CONFIDENCE
│         (Default: 0.60)               │
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 3: Student Exists in DB?         │ ── NO ──> REJECT: STUDENT_NOT_FOUND
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 4: Student Account Active?       │ ── NO ──> REJECT: STUDENT_INACTIVE
│         (is_active == 1)              │
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 5: Pass Record Exists?           │ ── NO ──> REJECT: NO_PASS
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 6: Pass Status == 'ACTIVE'?      │ ── NO ──> REJECT: PASS_INACTIVE
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 7: Date Validity Range Check     │ ── NO ──> REJECT: PASS_NOT_STARTED / PASS_EXPIRED
│         start <= current <= expiry    │
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 8: Route Authorization Check     │ ── NO ──> REJECT: ROUTE_MISMATCH
│         Pass Route == Bus Route       │
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ Rule 9: 5-Minute Cooldown Check       │ ── NO ──> REJECT: DUPLICATE_COOLDOWN
│         Elapsed >= 300 seconds        │
└───────────────────────────────────────┘
                  │ YES
                  ▼
┌───────────────────────────────────────┐
│ RESULT: ALLOWED (PASS_VALID)          │ ──> Log exactly one APPROVED record
└───────────────────────────────────────┘
```

---

## 5. Anti-Duplicate Boarding & Boundary Analysis

### Cooldown Logic:
- Configured via `cooldown_seconds: int = 300` (5 minutes).
- Grounded directly in SQLite `entry_logs` table (queries the latest `APPROVED` entry for the student on the same bus route).
- **Persistence**: Reboots or restarts of the application do not wipe the cooldown state.

### Explicit Boundary Behavior (Verified in Tests):
Initial Boarding at `10:00:00`:
- **At 10:04:59** (299 seconds elapsed): $\text{Elapsed} < 300 \rightarrow$ **REJECTED (`DUPLICATE_COOLDOWN`)**
- **At 10:05:00** (300 seconds elapsed): $\text{Elapsed} \ge 300 \rightarrow$ **ALLOWED (`PASS_VALID`)**
- **At 10:05:01** (301 seconds elapsed): $\text{Elapsed} \ge 300 \rightarrow$ **ALLOWED (`PASS_VALID`)**

### Date Expiration Convention:
- Passes use an **inclusive end-of-day** convention.
- A pass expiring on `2026-10-03` is valid through `23:59:59` on October 3rd, and expires on `2026-10-04 00:00:00`.

---

## 6. Test Suite & Coverage

The test suite in [`tests/test_pass_verifier.py`](file:///d:/SmartBusFaceRecognition/tests/test_pass_verifier.py) uses an isolated temporary SQLite database fixture and covers all 20 required scenarios:
1. `test_unknown_face_rejected`: Unknown person rejected.
2. `test_student_not_found`: Non-existent roll number rejected.
3. `test_no_pass_issued`: Active student without pass rejected.
4. `test_inactive_pass`: Suspended / revoked pass rejected.
5. `test_pass_not_started_yet`: Future start date rejected.
6. `test_pass_expired`: Past expiration date rejected.
7. `test_valid_pass_allowed`: Valid pass authorized.
8. `test_wrong_route_mismatch`: Route mismatch rejected.
9. `test_similarity_below_threshold`: Similarity < 0.60 rejected.
10. `test_similarity_above_threshold`: Similarity >= 0.60 allowed.
11. `test_first_boarding_allowed`: Initial scan permitted.
12. `test_cooldown_within_5_minutes_rejected`: Duplicate scan at 2 minutes rejected.
13. `test_cooldown_after_5_minutes_allowed`: Scan at 6 minutes permitted.
14. `test_different_bus_independent_cooldown`: Route 1 cooldown does not lock Route 2.
15. `test_different_student_same_bus_allowed`: Other students can board immediately.
16. `test_cooldown_persists_across_verifier_instances`: New verifier instance on same DB respects cooldown.
17. `test_missing_database_file_handled`: Corrupted / missing path returns `DATABASE_ERROR` without crashing.
18. `test_cooldown_exact_boundaries`: Evaluates 299s (Denied), 300s (Allowed), 301s (Allowed).
19. `test_inclusive_expiration_and_timezone_handling`: Inclusive expiration up to 23:59:59 verified.
20. `test_valid_pass_allowed`: Confirms correct student name and roll resolution.

---

## 7. Demo Script Execution Walkthrough

The standalone script [`src/pass_verifier_demo.py`](file:///d:/SmartBusFaceRecognition/src/pass_verifier_demo.py) validates the 5 canonical cases using seeded data:
- **Case 1**: Aarav Sharma (`21CS101`) on Bus 12 / Route R-101 $\rightarrow$ 🟢 **ALLOWED** (`PASS_VALID`)
- **Case 2**: Priya Patel (`21EC202`) with expired pass $\rightarrow$ 🔴 **DENIED** (`PASS_EXPIRED`)
- **Case 3**: Rohan Gupta (`21ME303`) with Route R-102 pass on Bus 12 $\rightarrow$ 🔴 **DENIED** (`ROUTE_MISMATCH`)
- **Case 4**: Unknown face (similarity 0.25) $\rightarrow$ 🔴 **DENIED** (`UNKNOWN_PERSON`)
- **Case 5**: Aarav Sharma re-scanning 2 minutes after Case 1 $\rightarrow$ 🔴 **DENIED** (`DUPLICATE_COOLDOWN`)

---

## 8. Remaining Limitations & Next Steps

- **Scope Adherence**: In accordance with user instructions, no UI (Streamlit), hardware webcam integration, or live camera looping was introduced in this stage.
- **Ready for Stage 5**:
  - Building the **Student Registration UI** in Streamlit (`app/pages/1_Register_Student.py`) to enroll students, capture 3-5 face samples with the webcam, generate their average 512-D embedding, and issue their bus pass.
