# Stage 7 Report: Real-Time Bus Entry Recognition
## AI-Based Face Recognition and Smart Bus Pass Verification System

---

## 1. Executive Summary

Stage 7 implements the **Real-Time Bus Entry Recognition Terminal** (`src/bus_entry_camera.py`) and its underlying orchestration engine (`src/core/bus_entry_service.py`). 

This module connects live camera input to the deep-learning vision models, checks passenger eligibility against the SQLite database using business rules, enforces the 5-minute anti-duplicate cooldown, and displays distinct color-coded visual feedback on the camera stream:
- 🟢 **GREEN**: ENTRY ALLOWED
- 🔴 **RED**: ENTRY DENIED (with specific reason)
- 🟡 **AMBER**: UNKNOWN PERSON

All 10 automated orchestration tests passed, and the live camera application was verified on the webcam hardware.

---

## 2. Files Created & Modified

| File | Purpose | Status |
| :--- | :--- | :---: |
| `src/core/bus_entry_service.py` | Orchestration service connecting camera frames, MTCNN, FaceNet, and PassVerifier | **Created** |
| `src/core/__init__.py` | Exported `BusEntryService`, `RecognitionDetail`, `BusEntryResult` | **Modified** |
| `src/bus_entry_camera.py` | Live webcam boarding terminal with HUD display, bounding boxes, and error recovery | **Created** |
| `tests/test_bus_entry_service.py` | 10 automated unit tests covering all boarding, denial, unknown, and DB failure cases | **Created** |
| `docs/live_bus_entry.md` | Comprehensive real-time pipeline documentation and troubleshooting guide | **Created** |
| `TODO.md` | Updated roadmap tracking Stage 7 completion | **Modified** |

---

## 3. Existing Modules Reused Without Modification

In strict accordance with the instructions, zero existing working modules were rewritten:
1. **`src/vision/face_detector.py` (`FaceDetector`)**: Reused MTCNN detection, bounding box clamping, and 160x160 cropping.
2. **`src/vision/face_recognizer.py` (`FaceRecognizer` & `compare_embeddings`)**: Reused InceptionResnetV1 512-D feature extraction and cosine similarity metric.
3. **`src/vision/embedding_store.py` (`EmbeddingStore`)**: Reused file-backed biometric persistence (`models/embeddings.pkl`).
4. **`src/core/pass_verifier.py` (`PassVerifier` & `ReasonCode`)**: Reused 9-step business rules engine and SQLite logging.
5. **`src/database/schema.py` & `src/database/crud.py`**: Reused `students`, `bus_routes`, `bus_passes`, and `entry_logs` tables.

---

## 4. Configuration Settings

All terminal parameters are centralized at the top of [`src/bus_entry_camera.py`](file:///d:/SmartBusFaceRecognition/src/bus_entry_camera.py) and can be overridden via command-line flags:

| Setting | Default Value | CLI Flag | Description |
| :--- | :--- | :--- | :--- |
| `CURRENT_BUS_ID` | `"BUS-12"` | `--bus` | Identifier of current bus |
| `CURRENT_ROUTE` | `"R-101"` | `--route` | Identifier of current route |
| `RECOGNITION_INTERVAL` | `2` | `--interval` | Process neural network every $N$ frames for responsiveness |
| `RECOGNITION_THRESHOLD` | `0.60` | `--threshold` | Minimum cosine similarity required to identify a student |
| `DISPLAY_RESULT_SECONDS` | `2.5` | — | Display duration for status banners |
| `DEBUG_MODE` | `False` | `--debug` | Shows FPS, raw reason codes, and inference latencies |

---

## 5. Automated Tests Executed & Results

Executed via:
```bash
python tests/test_bus_entry_service.py
```

### Test Cases Summary (10/10 Passed in 0.552s):
1. `test_known_student_valid_pass_allowed`: Known student on correct route $\rightarrow$ **ALLOWED (PASS_VALID)**.
2. `test_known_student_expired_pass_denied`: Expired pass $\rightarrow$ **DENIED (PASS_EXPIRED)**.
3. `test_known_student_wrong_route_denied`: Route mismatch $\rightarrow$ **DENIED (ROUTE_MISMATCH)**.
4. `test_unknown_person_denied`: Unregistered face $\rightarrow$ **DENIED (UNKNOWN_PERSON)** without guessing.
5. `test_duplicate_cooldown_denied`: Second scan at 2 minutes $\rightarrow$ **DENIED (DUPLICATE_COOLDOWN)**.
6. `test_multiple_faces_processed_independently`: Single frame with Aarav (Allowed) and Priya (Denied) processed independently.
7. `test_no_faces_in_frame`: Empty frame returns `[]`.
8. `test_invalid_frame_handled_safely`: `None` or empty array returns `[]` without crashing.
9. `test_low_confidence_match_treated_as_unknown`: Similarity < 0.60 rejected as unknown.
10. `test_database_failure_does_not_crash_service`: SQLite exception caught gracefully, returns `DATABASE_ERROR` while stream continues.

---

## 6. Live Webcam Test Execution

Command executed:
```bash
python src/bus_entry_camera.py --frames 5 --headless --debug
```

### Actual Terminal Output:
```
====================================================================
  STAGE 7: REAL-TIME BUS ENTRY RECOGNITION CAMERA
  Operating on BUS: BUS-12 | ROUTE: R-101
====================================================================

[1/3] Initializing BusEntryService...
      Recognition threshold: 0.60
      Enrolled students in store: 0

[2/3] Opening Webcam (Camera Index 0)...

[3/3] Live camera feed running!
--------------------------------------------------------------------
Reached target limit of 5 frames. Exiting...
--------------------------------------------------------------------
Session complete: Processed 5 frames in 0.81s (6.2 FPS).
Camera hardware released cleanly.
```

---

## 7. Problems Encountered & Fixes Applied

1. **Circular Import / Runpy Warning on Direct Execution**:
   - *Problem*: Direct module execution (`python -m src.data.student_importer`) triggered a `RuntimeWarning: found in sys.modules prior to execution`.
   - *Fix*: Decoupled `src/data/__init__.py` to avoid premature submodule loading during direct CLI runs.
2. **Database Logging Resiliency in Camera Loop**:
   - *Problem*: If database writes fail (e.g. database locked by another process), an unhandled exception would crash the live camera stream.
   - *Fix*: Wrapped `verifier.log_verification()` in an isolated `try/except` block inside `BusEntryService.process_frame()`, logging a non-fatal warning so the camera keeps running smoothly.

---

## 8. Known Limitations & Next Steps

* **Current Limitations**:
  * Face detection requires reasonable front-facing orientation ($\pm 40^\circ$).
  * Adverse lighting (total darkness or direct backlight) can reduce MTCNN detection confidence.
* **Stop Condition**: In strict adherence to your instructions, development is stopped after Stage 7. Dashboard and deployment modules will not be started.
