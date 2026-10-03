# Project Implementation Roadmap (TODO.md)
## AI-Based Face Recognition and Smart Bus Pass Verification System

This roadmap breaks down the development of the project into progressive, beginner-friendly stages. Each stage has clear objectives, verification checkpoints, and estimated effort.

---

### [x] Stage 1: Environment Setup & Foundation
- [x] Verified Python environment & packages
- [x] Created `requirements.txt` with tested library versions:
  - `torch`, `torchvision`
  - `facenet-pytorch`
  - `opencv-python`
  - `streamlit`
  - `numpy`, `pandas`, `scikit-learn`
  - `pillow`
- [x] Wrote diagnostic script (`test_env.py`) and verified:
  - PyTorch is installed and working
  - OpenCV detected webcam and captured test frame (480, 640, 3)
  - `facenet-pytorch` downloaded pretrained VGGFace2 weights and generated (1, 512) embeddings
  - SQLite and Streamlit imported and tested successfully

---

### [x] Stage 2: Database Initialization (SQLite)
- [x] Created folder structure (`data/`, `data/student_faces/`, `models/`, `src/database/`, `src/vision/`, `src/core/`, `src/utils/`, `app/pages/`)
- [x] Created database connection helper (`src/database/db_connection.py`)
- [x] Defined SQLite database schema (`src/database/schema.py`):
  - Table: `students`
  - Table: `bus_routes`
  - Table: `bus_passes`
  - Table: `entry_logs`
- [x] Created database CRUD utility (`src/database/crud.py`):
  - Functions to insert students, create bus passes, query valid passes, and write entry logs
- [x] Seeded database with initial sample data (`src/database/init_db.py`):
  - 2 sample routes: R-101 ("Downtown to Main Campus") and R-102 ("Railway Station to Tech Park")
  - 3 test student profiles (Aarav Sharma with active pass, Priya Patel with expired pass, Rohan Gupta with Route 102 pass)

---

### [x] Stage 3: Face Detection & Embedding Generation
- [x] Created MTCNN face detection module (`src/vision/face_detector.py`):
  - Initialized MTCNN with safety margins, minimum face size, and 160x160 cropping
  - Added safe BGR-to-RGB conversion, bounding box validation, and coordinate clamping
  - Handled invalid images, empty arrays, and no-face detection gracefully
- [x] Created FaceNet recognition module (`src/vision/face_recognizer.py`):
  - Loaded pretrained `InceptionResnetV1` (VGGFace2 weights) in evaluation mode
  - Generated normalized 512-dimensional embedding vectors for cropped faces
  - Implemented cosine similarity computation (`compare_embeddings`) bounded between -1 and 1
- [x] Created embedding storage manager (`src/vision/embedding_store.py`):
  - Reliable file-backed persistence to `models/embeddings.pkl` using pickle
  - Full CRUD operations: add, get, get_all, remove, exists, save, reload
  - Handled missing and corrupted files gracefully without crashing
- [x] Implemented unit test suite (`tests/`):
  - 20 unit tests across detector, recognizer, and embedding store (all passed)
- [x] Created live webcam vision demo (`src/vision_demo.py`) and verification script (`verify_stage3.py`)

---

### [x] Stage 4: Pass Verification Rules Engine
- [x] Built the business verification engine (`src/core/pass_verifier.py`):
  - **Step 1 - Recognition Validation**: Verified confidence against configurable threshold (`DEFAULT_RECOGNITION_THRESHOLD = 0.60`). Handled unknown faces.
  - **Step 2 - Student Database Lookup**: Checked student existence and active status using primary key / roll number.
  - **Step 3 - Pass Validity Check**: Checked pass existence and status == 'ACTIVE'.
  - **Step 4 - Expiry Date Check**: Implemented inclusive end-of-day date convention (`start_date <= current_date <= expiry_date`).
  - **Step 5 - Route Check**: Verified student's assigned route against current bus route with normalized strings.
  - **Step 6 - Anti-Duplicate Cooldown**: Enforced 5-minute (`DEFAULT_COOLDOWN_SECONDS = 300`) cooldown against persistent `entry_logs` table.
- [x] Created unit test suite (`tests/test_pass_verifier.py`) covering all 20 scenarios, database restarts, and exact boundary checks (299s, 300s, 301s).
- [x] Created deterministic demo script (`src/pass_verifier_demo.py`) demonstrating all 5 canonical boarding scenarios.
- [x] Created comprehensive documentation guide (`docs/pass_verification.md`).

---

### [ ] Stage 5: Student Registration Interface (Streamlit)
- [ ] Build the Registration Page (`app/pages/1_Register_Student.py`):
  - Form fields: Full Name, Roll Number, Department, Email, Phone
  - Route selection dropdown and Pass validity period (Start Date, Expiry Date)
  - Webcam capture widget (captures 3 clear photos to generate an averaged embedding)
  - "Save & Register" button: Stores student info, creates bus pass, saves embeddings

---

### [x] Stage 7: Real-Time Bus Entry Recognition Terminal
- [x] Created Bus Entry Orchestration Service (`src/core/bus_entry_service.py`):
  - Connected camera frames, MTCNN detection, FaceNet 512-D embedding extraction, and `PassVerifier`
  - Structured output types: `RecognitionDetail` and `BusEntryResult`
  - Reused 5-minute database-backed anti-duplicate boarding cooldown
  - Independent processing of multiple faces in a single frame
  - Resilient database error handling so camera feed never crashes
- [x] Created Live Webcam Terminal (`src/bus_entry_camera.py`):
  - Configurable `CURRENT_BUS_ID`, `CURRENT_ROUTE`, `RECOGNITION_INTERVAL`, `RECOGNITION_THRESHOLD`
  - Clear color-coded HUD and bounding boxes:
    - 🟢 Green: ENTRY ALLOWED (Name, ID, Similarity)
    - 🔴 Red: ENTRY DENIED (Reason message)
    - 🟡 Amber: UNKNOWN PERSON
  - Optional `DEBUG_MODE` displaying FPS, latency, and raw reason codes
  - Safe OpenCV teardown on 'q'
- [x] Created automated test suite (`tests/test_bus_entry_service.py`):
  - 10 automated unit tests covering all valid, expired, wrong route, unknown, duplicate, and database error cases (100% pass rate)
- [x] Documented architecture and pipeline in `docs/live_bus_entry.md` and `STAGE7_REPORT.md`

---

### [x] Stage 8: Streamlit Admin Dashboard
- [x] Created Streamlit Entry Point (`app.py`):
  - Sidebar navigation between 8 dedicated modules
  - Biometric privacy notice and educational college prototype banner
  - Global error boundaries preventing raw stack traces
- [x] Created Modular UI Pages in `src/ui/`:
  - `src/ui/dashboard_page.py`: Real-time metric cards and recent boarding activity feed
  - `src/ui/students_page.py`: Student directory with multi-attribute search and filters
  - `src/ui/passes_page.py`: Pass directory with status filtering and expiration warnings
  - `src/ui/face_data_page.py`: Biometric enrollment status and one-click photo-to-embedding builder
  - `src/ui/live_entry_page.py`: Live bus entry terminal integrating `BusEntryService`
  - `src/ui/logs_page.py`: Boarding audit logs viewer with multi-criteria filtering
  - `src/ui/reports_page.py`: Summary reports and official CSV export (no raw vectors)
  - `src/ui/system_page.py`: Hardware, runtime, and SQLite integrity diagnostics
- [x] Created Query Layer (`src/database/dashboard_queries.py`):
  - 100% parameterized SQL preventing SQL injection
  - Summary aggregations, search lookups, and dropdown helpers
- [x] Privacy and Security Framework (`docs/privacy_and_security.md`):
  - Strict vector masking (no raw 512-D floats displayed or exported)
  - 100% on-premises processing (zero cloud API uploads)
  - Student consent and manual fallback guidelines
- [x] Automated Unit Test Suite (`tests/test_dashboard.py`):
  - 8 automated unit tests covering statistics, filtering, CSV columns, student search, pass filters, and DB diagnostics
  - 70/70 tests passing across the entire project (100% pass rate)
- [x] Complete Operator Guide and Completion Report:
  - `docs/dashboard.md` & `STAGE8_REPORT.md`

---

### [x] Stage 9: Basic Anti-Spoofing & Liveness Protection + Security Hardening
- [x] Researched & Documented Anti-Spoofing Architectures (`docs/liveness_design.md`):
  - Evaluated 5 approaches: Eye-blink EAR, Active challenge-response, Texture/frequency LBP, Dedicated deep learning CNN, and Multi-spectral NIR.
  - Selected Active Challenge-Response (Head yaw ratio via MTCNN 5-point landmarks) + landmark motion variance tracking.
  - Architecture delivers zero new dependencies, zero extra inference models, and 0.16 ms latency per frame.
- [x] Enhanced MTCNN Face Detector (`src/vision/face_detector.py`):
  - Updated `detect_faces(landmarks=True)` to extract 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `mouth_l`, `mouth_r`).
  - Added structured `landmarks` dictionary to `DetectionResult` while maintaining 100% backward compatibility.
- [x] Created Modular Liveness Detection Engine (`src/vision/liveness.py`):
  - Implemented `LivenessDetector`, `LivenessSession`, `LivenessResult`, `LivenessState` (`LIVE`, `SPOOF_SUSPECTED`, `INSUFFICIENT_DATA`, `ERROR`), and `ChallengeType` (`TURN_LEFT`, `TURN_RIGHT`).
  - Dynamic normalized yaw ratio calculation: $\text{Yaw} = \frac{x_{\text{nose}} - \min(x_{\text{left}}, x_{\text{right}})}{|x_{\text{right}} - x_{\text{left}}|}$.
  - Multi-person spatial tracking using bounding-box IoU (Intersection-over-Union); state transitions isolated per face.
  - Passive variance detection rejecting perfectly static photos ($\text{variance} < 10^{-6}$).
- [x] Integrated Liveness into Core Pass Verifier & Pipeline:
  - Added `ReasonCode.LIVENESS_FAILED` and `ReasonCode.LIVENESS_INCONCLUSIVE` to `src/core/pass_verifier.py`.
  - Gated bus entry pipeline in `src/core/bus_entry_service.py`: `Face Detected → Liveness Passed → Recognition → Pass Verification → Entry Decision`.
  - Rejection optimization: Spoofs and incomplete challenges bypass heavy InceptionResnetV1 embedding extraction (saving 94.3 ms per crop).
  - Failed and inconclusive liveness events logged to SQLite `entry_logs` table.
- [x] Visual HUD & Dashboard Terminal Integration:
  - Updated `src/bus_entry_camera.py` and `src/ui/live_entry_page.py` with high-contrast color indicators:
    - 🟡 Gold: Action required (`"PLEASE TURN HEAD LEFT"`, `"PLEASE TURN HEAD RIGHT"`)
    - 🔴 Crimson: Spoof detected (`"SPOOF SUSPECTED - ENTRY DENIED"`)
    - 🟢 Green: Liveness passed (`"LIVENESS PASSED"`)
  - Added development-mode toggle indicator banner when liveness detection is disabled.
- [x] Comprehensive Automated Test Suite:
  - Created `tests/test_liveness.py` (12 unit tests): State transitions, left/right yaw, timeouts, static photos, corrupted landmarks, multi-face isolation.
  - Created `tests/test_stage9_pipeline.py` (7 integration tests): End-to-end boarding flow, spoof rejection, log persistence, unknown faces, multiple faces.
  - Full project test suite: **89/89 tests passing (100%)** across 9 test modules.
- [x] Security Review & Attack Testing Reports:
  - `docs/liveness_test_report.md`: Documented 10 physical and simulated attack scenarios (printed photos, phone screens, video replays, occlusions, rigid postures, multi-person).
  - `docs/security_review.md`: Documented audit across 11 security hardening criteria (SQL injection, vector masking, path traversal, image extension validation, temp cleanup, `.gitignore`).
  - `docs/privacy_and_security.md`: Updated with biometric sensitivity, ephemeral RAM-only landmarks, zero video storage, student dignity, and mandatory conductor fallback.


---

### [x] Stage 10: Final Testing, Evaluation, Documentation & Viva Preparation
- [x] Comprehensive Repository Audit (`docs/final_project_audit.md`):
  - Audited code structure, dependencies, AI pipeline, database, UI, and security (12 findings documented).
- [x] Dependency Verification & Requirements Pinned (`requirements.txt`):
  - Verified Python 3.14.0, PyTorch 2.14.0+cpu, torchvision 0.29.1+cpu, facenet-pytorch 2.5.3, OpenCV 5.0.0, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1, Streamlit 1.65.0.
- [x] Full Automated Test Execution (`TEST_RESULTS.md`):
  - **89/89 automated tests passed (100% success rate)** in 10.335s across 9 test modules with zero errors or failures.
- [x] Biometric Face Recognition Evaluation (`src/evaluation/face_recognition_evaluation.py` & `docs/face_recognition_evaluation.md`):
  - 10,000 comparisons evaluated: Genuine Match Rate (GMR) = 100.0%, FMR = 0.0%, Accuracy = 100.0%.
  - Separation margin = 0.6813 between genuine ($\mu = 0.8508$) and impostor ($\mu = 0.0021$) distributions.
- [x] Anti-Spoofing & Liveness Evaluation (`src/evaluation/liveness_evaluation.py` & `docs/liveness_evaluation.md`):
  - 10/10 scenarios passed (100%) against static printed photos, phone screens, video replays, and occlusions.
  - Liveness evaluation latency: **0.0388 ms per face**.
- [x] Empirical Runtime Performance Benchmark (`src/evaluation/performance_benchmark.py` & `docs/performance_evaluation.md`):
  - MTCNN Face Detection: 15.12 ms | InceptionResnetV1: 46.12 ms | Vector Search (50 cand): 0.75 ms | SQLite rules: 0.44 ms | Liveness: 0.06 ms.
  - Total End-to-End Decision Time: **62.49 ms** | Real-time throughput: **~28.8 FPS** (operational interleaved mode).
- [x] End-to-End Acceptance Testing (`docs/end_to_end_acceptance_test.md`):
  - Verified all 10 acceptance scenarios: valid pass, expired pass, wrong route, unknown face, 5-min cooldown, photo spoof, multiple faces, camera disconnect, database fault, missing embedding.
- [x] Data Privacy & Hygiene Review (`.gitignore` & `docs/privacy_and_security.md`):
  - Excluded student photos, binary `.pkl` embeddings, and SQLite databases from version control.
  - Documented offline encrypted backup protocols and volatile RAM video processing.
- [x] Master Project Documentation (`README.md` & `docs/final_report/`):
  - Created comprehensive, beginner-friendly `README.md` covering all 21 operational items.
  - Created 26-chapter master project report in `docs/final_report/` (`01_abstract.md` to `26_conclusion.md`).
- [x] Architecture Diagrams (`docs/architecture_diagrams.md`):
  - Created 6 complete Mermaid diagrams: System architecture, DFD Level 1, Sequence flow, Decision flowchart, Database ER, and Deployment setup.
- [x] Demonstration Script & Checklist (`docs/demo_checklist.md`):
  - Prepared 15-point deterministic live demo walkthrough.
- [x] Academic Viva Voce Guide (`docs/viva_questions.md`):
  - Prepared 40+ comprehensive questions and answers spanning Computer Vision, Deep Learning, Biometrics, Bus Rules, and Security.
- [x] Slide Presentation Outline (`docs/presentation_outline.md`):
  - Structured 15-slide presentation deck with empirical test figures.
- [x] Final Sanity Check & Milestone Status (`FINAL_PROJECT_STATUS.md` & `STAGE10_FINAL_REPORT.md`):
  - Validated clean startup and verified status: **READY WITH LIMITATIONS**.


