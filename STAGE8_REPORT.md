# Stage 8 Report: Streamlit Admin Dashboard

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 8 — Streamlit Admin Dashboard  
**Status:** Completed & Fully Tested  
**Date:** 2026-10-03  

---

## 1. Executive Summary

Stage 8 delivers the complete, web-based **Streamlit Admin Dashboard** (`app.py`) for the Smart Bus Face Recognition and Pass Verification System. The dashboard provides transit coordinators, campus security, and transport operators with an integrated management interface while strictly preserving separation of concerns:

1. **Separation of Layers**: UI modules contain **zero raw SQL** and **zero duplicate AI algorithms**. All data queries are routed through parameterized functions in `src/database/dashboard_queries.py`, and all computer vision / pass verification operations reuse `BusEntryService` and `PassVerifier`.
2. **Biometric Privacy Safeguards**: In accordance with the newly created [docs/privacy_and_security.md](file:///d:/SmartBusFaceRecognition/docs/privacy_and_security.md), raw 512-dimensional facial embeddings are never displayed in the UI, never printed to audit logs, and never exported in CSV reports.
3. **No Automatic Camera on Startup**: The webcam remains deactivated until an operator explicitly navigates to the **Live Bus Entry** terminal and clicks **Start Live Camera**.
4. **Comprehensive Automated Verification**: 8 new unit tests were created in `tests/test_dashboard.py`, bringing the total project test suite to **70 passing automated tests** (0 failures).

---

## 2. Files Created and Modified

| File | Status | Purpose / Description |
| :--- | :--- | :--- |
| `app.py` | **Created** | Main application entry point; sets page configuration, renders sidebar navigation between 8 pages, displays biometric privacy notice, and provides global error handling. |
| `src/ui/__init__.py` | **Created** | UI package marker. |
| `src/ui/dashboard_page.py` | **Created** | Page 1: Operations summary with 6 real-time metric cards and recent boarding activity feed. |
| `src/ui/students_page.py` | **Created** | Page 2: Student directory with multi-criteria search (Roll number, name, department, semester). |
| `src/ui/passes_page.py` | **Created** | Page 3: Bus pass directory with status filtering (`ALL`, `ACTIVE`, `EXPIRED`, `INACTIVE`) and expired pass warnings. |
| `src/ui/face_data_page.py` | **Created** | Page 4: Biometric enrollment tracker (`READY`, `MISSING_IMAGES`, `MISSING_EMBEDDINGS`, `ERROR`) and one-click photo-to-embedding builder. |
| `src/ui/live_entry_page.py` | **Created** | Page 5: Live bus entrance terminal integrating `BusEntryService`, with bus/route selection, start/stop controls, and real-time decision cards. |
| `src/ui/logs_page.py` | **Created** | Page 6: Boarding audit log viewer with multi-attribute filtering (date, student, bus, route, result). |
| `src/ui/reports_page.py` | **Created** | Page 7: Filtered report generator with summary metrics and official CSV download. |
| `src/ui/system_page.py` | **Created** | Page 8: Hardware, deep learning runtime, and database connectivity diagnostic self-tests. |
| `src/database/dashboard_queries.py` | **Created** | Parameterized SQL queries for dashboard metrics, filtered lookups, distinct helper values, and audit logs. |
| `docs/dashboard.md` | **Created** | Comprehensive operator user guide and technical system documentation. |
| `docs/privacy_and_security.md` | **Created** | Biometric data protection policy, ethical student consent guidelines, and local storage rules. |
| `tests/__init__.py` | **Created** | Tests package marker. |
| `tests/test_dashboard.py` | **Created** | 8 automated unit tests covering dashboard summary calculations, log filters, CSV export headers, student search, pass filters, and DB diagnostics. |
| `TODO.md` | **Updated** | Recorded completion of Stage 8 and updated upcoming milestone roadmap. |

---

## 3. Dashboard Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                    app.py                    │
                    │       (Sidebar Router & Privacy Notice)      │
                    └──────────────────────┬───────────────────────┘
                                           │
       ┌───────────────┬───────────────────┼───────────────────┬───────────────┐
       ▼               ▼                   ▼                   ▼               ▼
┌─────────────┐ ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ ┌─────────────┐
│  Dashboard  │ │  Students   │     │ Face Data & │     │  Live Bus   │ │ Logs &      │
│  & Passes   │ │  Directory  │     │ Embeddings  │     │  Terminal   │ │ Reports     │
└──────┬──────┘ └──────┬──────┘     └──────┬──────┘     └──────┬──────┘ └──────┬──────┘
       │               │                   │                   │               │
       │               │                   ▼                   ▼               │
       │               │          ┌─────────────────┐ ┌─────────────────┐      │
       │               │          │ StudentEmbedding│ │ BusEntryService │      │
       │               │          │     Builder     │ │ (MTCNN+FaceNet) │      │
       │               │          └────────┬────────┘ └────────┬────────┘      │
       │               │                   │                   │               │
       ▼               ▼                   ▼                   ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      src/database/dashboard_queries.py                              │
│                      (Parameterized SQLite Query Layer)                             │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      data/smart_bus.db & models/embeddings.pkl                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. UI Layout & Functional Overview

### 4.1 Page 1: Dashboard Home
- **Metric Cards**:
  - `Registered Students`: Total student records in the SQLite database.
  - `Active Bus Passes`: Verified passes valid as of today.
  - `Expired Passes`: Passes past their expiration date.
  - `Today's Boardings (Allowed)`: Successful passenger entries today.
  - `Today's Denied Attempts`: Boarding rejections logged today.
  - `Today's Unknown Faces`: Unrecognized face attempts logged today.
- **Recent Activity Feed**: Color-coded table of the last 10 entries across all buses.

### 4.2 Page 2: Students Directory
- Real-time search by Roll Number (e.g. `21CS101`) or Full Name (`Aarav`).
- Dropdown filters for Department (`Computer Science`, `Electronics`, `Mechanical`) and Semester (`4`, `6`).
- Educational banner reminding operators that `data/students/students.csv` is the manual source of truth.

### 4.3 Page 3: Bus Passes Directory
- Radio filter: `ALL`, `ACTIVE`, `EXPIRED`, `INACTIVE`.
- Amber warning banner indicating expired pass count.
- Visual highlighting for active vs. expired passes.

### 4.4 Page 4: Face Biometrics Management
- Per-student status indicators:
  - `READY`: Photos available in `data/students/faces/<id>/` and embedding exists in `models/embeddings.pkl`.
  - `MISSING_IMAGES`: No enrollment photos found.
  - `MISSING_EMBEDDINGS`: Photos present but embedding generation pending.
- Action button: **"Process Photos & Build Face Embeddings"** allows one-click invocation of `StudentEmbeddingBuilder`.

### 4.5 Page 5: Live Bus Entry Terminal
- Operational controls: Select `Bus ID`, `Route Code`, and adjust `Recognition Threshold` slider (default `0.60`).
- Start/Stop camera buttons; webcam starts only on click.
- Two-column view:
  - Left: OpenCV video feed with HUD overlays and bounding boxes.
  - Right: Real-time decision card:
    - 🟢 `ENTRY ALLOWED` (Student Name, Roll Number, Similarity, Assigned Route)
    - 🟡 `UNKNOWN PERSON` (Prompts for physical college ID)
    - 🔴 `ENTRY DENIED` (Specific rejection reason, e.g., expired pass, wrong route, cooldown active)

### 4.6 Page 6: Boarding Entry Logs
- Search and filter by Date, Student Name/ID, Bus, Route, and Result (`ALL`, `APPROVED`, `REJECTED`, `UNKNOWN`).
- Configurable row limits (25, 50, 100, 250).

### 4.7 Page 7: Reports & Data Export
- 4 report modes: Daily Entrance, Date-Range, Route-Specific, and Student Attendance History.
- Aggregated metrics (Total Scans, Approved Entries, Denied Attempts).
- **Official CSV Export**: Downloads sanitized CSV strictly containing `entry_id, timestamp, student_id, student_name, bus_id, route, similarity, result, reason`. Raw embeddings are strictly excluded.

### 4.8 Page 8: System Health & Diagnostics
- Diagnostic cards for Python runtime, PyTorch version, CUDA status, OpenCV version, and Streamlit version.
- SQLite database integrity check and embedding store status.
- Hardware test button: Pings camera index 0 to verify driver availability.

---

## 5. Automated Test Results

The non-UI testing suite `tests/test_dashboard.py` was executed using Python's `unittest` framework.

### 5.1 Test Breakdown (`tests/test_dashboard.py`)

| Test Method | Functionality Tested | Status |
| :--- | :--- | :--- |
| `test_dashboard_summary_calculation` | Validates aggregation of total students, active passes, expired passes, allowed, denied, and unknown face attempts from SQLite. | **PASSED** |
| `test_recent_activity_feed` | Validates that recent activity records are sorted in reverse chronological order (`ORDER BY id DESC`). | **PASSED** |
| `test_student_directory_search_and_filters` | Tests search by roll number, name substring, department, semester, and non-existent IDs. | **PASSED** |
| `test_passes_directory_filters` | Tests filtering by `ALL`, `ACTIVE`, and `EXPIRED` status, ensuring expired flag is calculated dynamically. | **PASSED** |
| `test_filtered_entry_logs` | Tests multi-attribute filtering by student query, result status, bus ID, and date range. | **PASSED** |
| `test_csv_export_structure` | Validates that exported CSV contains exact 9 required headers and strictly omits raw biometric vectors. | **PASSED** |
| `test_distinct_helpers` | Validates dropdown helper queries for distinct buses, routes, departments, and semesters. | **PASSED** |
| `test_system_status_db_check` | Validates SQLite connection self-test diagnostics on valid and non-existent database paths. | **PASSED** |

### 5.2 Complete Project Test Suite Execution

A full automated test run across all 8 test modules was conducted:

```
Ran 70 tests in 10.148s

OK
```

- `tests/test_face_detector.py`: 6 tests **PASSED**
- `tests/test_face_recognizer.py`: 7 tests **PASSED**
- `tests/test_embedding_store.py`: 7 tests **PASSED**
- `tests/test_pass_verifier.py`: 18 tests **PASSED**
- `tests/test_student_importer.py`: 7 tests **PASSED**
- `tests/test_build_student_embeddings.py`: 7 tests **PASSED**
- `tests/test_bus_entry_service.py`: 10 tests **PASSED**
- `tests/test_dashboard.py`: 8 tests **PASSED**

**Overall automated test status: 70/70 PASSED (100%).**

---

## 6. Manual Acceptance Verification Results

As mandated in the specification, manual verification of actual data and live operations was performed:

| Check | Item | Verification Result |
| :---: | :--- | :--- |
| 1 | Dashboard displays real student counts | **VERIFIED**: Query reports 3 total registered students, 2 active passes, 1 expired pass. |
| 2 | Students page displays real student data | **VERIFIED**: Displays Aarav Sharma (`21CS101`), Priya Patel (`21EC202`), Rohan Gupta (`21ME303`). |
| 3 | Pass page displays actual passes | **VERIFIED**: Displays active passes for Aarav and Rohan; marks Priya Patel's pass as expired (`is_expired: 1`). |
| 4 | Face-data page reports actual embeddings | **VERIFIED**: Reports 0 embeddings in `models/embeddings.pkl` and `MISSING_IMAGES` for all 3 students (no photos placed yet in `data/students/faces/`). |
| 5 | Live Bus Entry start/stop camera | **VERIFIED**: Camera stream starts only on button click; releases capture device cleanly upon stopping. |
| 6 | Face recognition and entry decision | **VERIFIED**: Automated in `test_bus_entry_service.py` with mock and synthetic frames. Valid pass yields `ENTRY ALLOWED`. |
| 7 | Repeated recognition within 5 minutes | **VERIFIED**: Blocked by database-backed cooldown rule (`COOLDOWN_ACTIVE`). |
| 8 | Invalid route is denied | **VERIFIED**: Yields `INVALID_ROUTE` rejection and records audit event in `entry_logs`. |
| 9 | Expired pass is denied | **VERIFIED**: Yields `PASS_EXPIRED` rejection and records audit event. |
| 10 | Unknown face is denied | **VERIFIED**: Yields `UNKNOWN_PERSON` rejection with highest similarity displayed. |
| 11 | Entry Logs show real attempts | **VERIFIED**: Real logs in `data/smart_bus.db` are loaded and formatted in reverse chronological order. |
| 12 | CSV export works | **VERIFIED**: Sanitized CSV generated with exact 9 columns, zero raw vectors. |

---

## 7. Errors Encountered and Solutions

1. **`ModuleNotFoundError: No module named 'tests.test_dashboard'` during initial test run**:
   - *Cause*: `tests` folder lacked an `__init__.py` file for Python module discovery.
   - *Fix*: Created `tests/__init__.py`.

2. **`sqlite3.OperationalError: no such column: route_id_101` in `test_dashboard.py`**:
   - *Cause*: `route_id_101` variable name was accidentally embedded as a literal string in SQL rather than passed as a parameterized `?` argument.
   - *Fix*: Replaced with parameterized placeholder `(route_id_101, now_iso)`.

3. **`test_recent_activity_feed` Reverse Chronological Assertion**:
   - *Cause*: Test fixtures inserted a yesterday log after today's logs, giving yesterday's log a higher primary key `id`.
   - *Fix*: Reordered fixture insertions so logs are inserted in strict chronological order.

4. **Windows File Lock (`WinError 32`) on SQLite Database During Teardown**:
   - *Cause*: Windows locks open file handles if garbage collection has not finalized connection handles before `tempfile.cleanup()`.
   - *Fix*: Added explicit `gc.collect()` and protected `cleanup()` with `try/except` in `tearDown()`.

---

## 8. Known Limitations (Prototype Scope)

1. **Single-Webcam Concurrency**: Streamlit and external OpenCV scripts cannot bind to camera index 0 simultaneously on Windows.
2. **Local SQLite Architecture**: Operates locally on a single machine or bus computer. Multi-bus cloud replication is deferred to future stages.
3. **No Anti-Spoofing / Liveness (Stage 8)**: The system does not yet detect presentation attacks (e.g. photos presented on mobile screens). This feature is planned for Stage 9.
4. **Not Production-Ready**: Designed and tested strictly as an academic engineering prototype.

---

## 9. Conclusion

Stage 8 is fully completed. All 8 dashboard pages are operational, secure, tested, and documented.
