# Chapter 15: Bus Pass Verification Rules Engine

## 15.1 Architectural Role of `PassVerifier`
The `PassVerifier` class (`src/core/pass_verifier.py`) acts as the deterministic gatekeeper of the transit system. It bridges the probabilistic output of the deep learning vision pipeline with the rigid legal and business rules stored in SQLite.

## 15.2 Sequential Rule Hierarchy
When a face match assertion is received, `PassVerifier` evaluates rules in a strict, non-bypassable sequence:

```
[ Recognition Result (student_id, similarity) ]
                     │
                     ▼
       Rule 1: Is Similarity >= 0.60? ──────────────> NO ──> REJECT (LOW_CONFIDENCE / UNKNOWN)
                     │ YES
                     ▼
       Rule 2: Does Student Exist in DB? ───────────> NO ──> REJECT (STUDENT_NOT_FOUND)
                     │ YES
                     ▼
       Rule 3: Is Student Account Active? ──────────> NO ──> REJECT (STUDENT_INACTIVE)
                     │ YES
                     ▼
       Rule 4: Does Student Have a Bus Pass? ───────> NO ──> REJECT (NO_PASS)
                     │ YES
                     ▼
       Rule 5: Is Pass Status == 'ACTIVE'? ─────────> NO ──> REJECT (PASS_INACTIVE)
                     │ YES
                     ▼
       Rule 6: Has Pass Started Yet? ───────────────> NO ──> REJECT (PASS_NOT_STARTED)
                     │ YES
                     ▼
       Rule 7: Has Pass Expired? ───────────────────> NO ──> REJECT (PASS_EXPIRED)
                     │ YES
                     ▼
       Rule 8: Does Route Match Assigned Route? ────> NO ──> REJECT (ROUTE_MISMATCH)
                     │ YES
                     ▼
       Rule 9: Has Student Boarded in Last 5 Min? ──> YES ─> REJECT (DUPLICATE_COOLDOWN)
                     │ NO
                     ▼
               AUTHORIZE BOARDING
           (Green Banner / PASS_VALID)
```

## 15.3 Inclusive Date Boundary Conventions
College semester passes expire on a specified date (e.g. `2026-12-31`). `PassVerifier` treats expiration dates **inclusively until the end of the day (23:59:59 UTC)**, ensuring that a student whose pass expires on December 31st is not prematurely denied entry on their afternoon return trip home on that date.
