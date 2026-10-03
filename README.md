# 🚌 AI-Based Face Recognition and Smart Bus Pass Verification System

An automated, privacy-first, on-device facial recognition and digital bus-pass verification platform designed for educational college transit.

---

## 1. Project Title & Overview
**AI-Based Face Recognition and Smart Bus Pass Verification System**  
A modular college engineering prototype that combines real-time deep learning computer vision (MTCNN + FaceNet InceptionResnetV1), active challenge-response liveness detection (anti-spoofing), and an SQLite-backed business rules engine to verify student bus passes seamlessly at the bus entrance door.

---

## 2. Problem Statement
Traditional college bus transit relies on physical paper cards or laminated plastic badges. This manual process causes multiple problems:
- **Boarding Congestion:** Conductors must manually inspect every pass during peak morning rushes, causing long boarding queues.
- **Pass Sharing & Fraud:** Students loan valid passes to non-eligible peers or siblings.
- **Expired & Wrong-Route Boarding:** Expired passes or incorrect bus routes go unnoticed due to human inspection fatigue.
- **Lack of Attendance Data:** Transport departments lack computerized audit logs of when and where students board.

---

## 3. Proposed Solution
This system replaces manual card checking with a contactless vision terminal at the bus entrance:
1. As a passenger enters, the entrance webcam detects their face.
2. The system issues an active head-movement challenge (e.g., "Please turn your head slightly LEFT") and tracks landmark motion to prevent 2D photo spoofing attacks.
3. Once liveness is verified, the system extracts a 512-dimensional facial feature embedding and matches it against enrolled students.
4. The business rules engine looks up the student in SQLite, checks active pass status, verifies the valid date window, confirms route authorization, and checks against a 5-minute duplicate boarding cooldown.
5. Instant color-coded visual feedback is displayed on the conductor's terminal screen, and the transaction is immutably logged.

---

## 4. Key Features
- **Deep Biometric Face Recognition:** Pretrained InceptionResnetV1 (VGGFace2) extracts normalized 512-D embeddings with high intra-class clustering.
- **Active Challenge-Response Anti-Spoofing:** Evaluates horizontal head yaw symmetry ($\psi$) from MTCNN landmarks to reject printed photos and phone screen displays without requiring extra deep models.
- **Sequential Business Rules Engine (`PassVerifier`):** Enforces 6 sequential validations (Identity $\rightarrow$ Student Status $\rightarrow$ Pass Active $\rightarrow$ Date Range $\rightarrow$ Route Match $\rightarrow$ 5-Min Cooldown).
- **Anti-Duplicate Boarding Protection:** Rejects duplicate scans within 300 seconds (5 minutes) to stop pass sharing.
- **Streamlit Web Admin Dashboard:** 8 dedicated management tabs for student directories, pass statuses, log exploration, report exports, and system diagnostics.
- **Privacy by Design & On-Device AI:** 100% on-premises execution; zero cloud uploads, ephemeral RAM video streaming, and strict masking of raw biometric vectors.
- **Mandatory Manual Fallback:** Non-punitive procedure allowing conductors to check physical college ID cards if facial recognition or liveness fails.

---

## 5. System Architecture

```
[ Passenger at Bus Door ] ──> [ Live HD Webcam ]
                                    │
                                    ▼
                      [ MTCNN Face Detector ]
                    (160x160 Crop + 5 Landmarks)
                                    │
                                    ▼
                       [ Liveness Detector ]
                     (Active Yaw Challenge + Motion)
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
            [ Liveness Failed ]           [ Liveness Passed ]
          (Crimson Box / DENIED)                  │
                                                  ▼
                                      [ FaceNet Recognizer ]
                                      (512-D Feature Vector)
                                                  │
                                                  ▼
                                      [ Embedding Store ]
                                  (Cosine Similarity >= 0.60)
                                                  │
                                                  ▼
                                        [ Pass Verifier ]
                                  (SQLite Rules & 5-Min Cooldown)
                                                  │
                                                  ▼
                                     [ Terminal HUD Banner ]
                               (Green: ALLOWED | Red: DENIED)
                                                  │
                                                  ▼
                                       [ SQLite Audit Log ]
```

---

## 6. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.9 – 3.14 | Core application programming |
| **Deep Learning Framework** | PyTorch & TorchVision (CPU Mode) | Neural network inference runtime |
| **Face Detection & Alignment**| MTCNN (`facenet-pytorch`) | Fast multi-task cascaded face localization |
| **Feature Extraction** | InceptionResnetV1 (`facenet-pytorch`) | 512-dimensional deep facial embeddings |
| **Computer Vision & Video** | OpenCV (`cv2`) & Pillow | Video streaming, preprocessing, HUD rendering |
| **Database Engine** | SQLite 3 | Zero-configuration transactional data storage |
| **Web Dashboard** | Streamlit | Responsive browser-based administrative interface |
| **Data Analytics** | NumPy, Pandas, Scikit-Learn | Vector math, data filtering, and reporting |

---

## 7. Hardware & Software Requirements

### Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Multi-core x86_64 CPU).
- **Memory:** 4 GB RAM minimum (8 GB recommended).
- **Camera:** Standard 720p or 1080p USB webcam or integrated laptop camera.
- **Storage:** At least 2 GB of available disk space (for PyTorch weights and database).

### Software Requirements
- **OS:** Windows 10/11, Ubuntu Linux 20.04+, or macOS.
- **Python:** Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14.

---

## 8. Installation & Setup Instructions

### Step 1: Clone or Open the Repository
```bash
cd SmartBusFaceRecognition
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify the Environment
```bash
python test_env.py
```
*Expected Output:* `ALL CHECKS PASSED! Ready for Stage 2.`

---

## 9. Project Directory Structure

```
SmartBusFaceRecognition/
├── app.py                             # Main Streamlit dashboard entry point
├── requirements.txt                   # Tested package dependencies
├── TODO.md                            # Complete milestone roadmap
├── README.md                          # Comprehensive user manual
├── .gitignore                         # Biometric & PII privacy exclusions
├── data/
│   ├── smart_bus.db                   # SQLite database (students, passes, logs)
│   └── students/
│       ├── students.csv               # Master student records import source
│       ├── students_template.csv      # Template for entering new student data
│       └── faces/                     # Face image folders (<student_id>/)
├── models/
│   └── embeddings.pkl                 # Serialized 512-D FaceNet embeddings
├── src/
│   ├── config.py                      # Global configuration constants
│   ├── bus_entry_camera.py            # Live OpenCV webcam entrance application
│   ├── core/
│   │   ├── bus_entry_service.py       # Orchestration pipeline
│   │   ├── config.py                  # Core verification constants
│   │   └── pass_verifier.py           # Pass rules, route & cooldown engine
│   ├── data/
│   │   ├── student_importer.py        # Validates & imports CSV data into SQLite
│   │   └── validate_real_data.py      # Data validation utility
│   ├── database/
│   │   ├── crud.py                    # Database queries & transactions
│   │   ├── dashboard_queries.py       # Analytical SQL queries for Streamlit
│   │   ├── db_connection.py           # Thread-safe SQLite connection manager
│   │   ├── init_db.py                 # Schema seeder
│   │   └── schema.py                  # Database table definitions
│   ├── evaluation/
│   │   ├── face_recognition_evaluation.py # Biometric accuracy & threshold sweep
│   │   ├── liveness_evaluation.py     # Anti-spoofing empirical evaluation
│   │   └── performance_benchmark.py   # Latency & FPS benchmarks
│   ├── ui/                            # Modular Streamlit dashboard pages
│   └── vision/
│       ├── build_student_embeddings.py# Photo-to-embedding generator
│       ├── embedding_store.py         # Vector similarity search & persistence
│       ├── face_detector.py           # MTCNN detector & landmark extractor
│       ├── face_recognizer.py         # InceptionResnetV1 feature extractor
│       └── liveness.py                # Active challenge-response detector
├── tests/                             # Complete automated test suite (89 tests)
└── docs/                              # Technical guides, reports & architecture
```

---

## 10. Manual Student Data Setup

1. Open `data/students/students.csv` in Excel, VS Code, or Notepad.
2. Enter student records using the standard schema:
```csv
student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,Computer Science,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
21EC202,Priya Patel,Electronics,6,BUS-12,R-101,2026-01-01,2026-08-31,ACTIVE
21ME303,Rohan Gupta,Mechanical,6,BUS-12,R-102,2026-07-01,2026-12-31,ACTIVE
```
3. Import the CSV records into SQLite:
```bash
python -m src.data.student_importer
```

---

## 11. Face Image Setup

To register a student for face recognition:
1. Create a folder named exactly after the student's Roll Number:
   ```
   data/students/faces/<student_id>/
   ```
   *Example:* `data/students/faces/21CS101/`
2. Place 3 to 5 clear frontal photos (`.jpg`, `.jpeg`, or `.png`) inside that folder:
   - `21CS101/photo1.jpg` (Frontal view)
   - `21CS101/photo2.jpg` (Slight head tilt)
   - `21CS101/photo3.jpg` (Different ambient lighting)

---

## 12. Generating Face Embeddings

Once photos are placed into the respective student directories, run:
```bash
python -m src.vision.build_student_embeddings
```
*What this does:*
1. Locates all student image folders.
2. Detects the face crop using MTCNN.
3. Generates 512-dimensional vectors with InceptionResnetV1.
4. Averages the vectors per student to produce a robust reference profile.
5. Saves the embeddings securely into `models/embeddings.pkl`.

*Note:* You can also trigger this with a single click from the **Face Data** page in the Streamlit Dashboard!

---

## 13. Starting the Real-Time Bus Camera Terminal

To start the entrance camera terminal on the bus:
```bash
python src/bus_entry_camera.py --bus BUS-12 --route R-101
```

### Available Command-Line Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--bus` | Bus ID assigned to this terminal | `BUS-12` |
| `--route` | Current active route code | `R-101` |
| `--camera` | Web camera device index (0, 1, 2) | `0` |
| `--threshold` | Cosine similarity recognition threshold | `0.60` |
| `--interval` | Run full neural recognition every N frames | `2` |
| `--debug` | Render live FPS, latencies, and reason codes | `False` |

*To exit the camera application, focus on the video window and press `q`.*

---

## 14. Starting the Streamlit Admin Dashboard

To start the browser-based administration portal:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Dashboard Tabs:
1. **📊 Dashboard:** Overview metrics, active passes, and real-time boarding feed.
2. **🎓 Students:** Filterable student directory with instant keyword search.
3. **🎫 Bus Passes:** Pass directory with active/expired status badges.
4. **👤 Face Data:** Biometric enrollment tracker and one-click embedding builder.
5. **📹 Live Bus Entry:** Embedded web camera terminal for testing boarding events.
6. **📝 Entry Logs:** Full audit history of accepted and denied attempts.
7. **📈 Reports:** Aggregated charts and official sanitized CSV report downloads.
8. **⚙️ System Status:** Hardware diagnostics and SQLite integrity monitor.

---

## 15. Running the Automated Test Suite

To execute all 89 automated tests:
```bash
python -m unittest discover tests/ -v
```
*Expected Output:*
```
Ran 89 tests in 10.335s
OK
```

---

## 16. Running Performance & Biometric Evaluations

### Face Recognition Biometric Accuracy Evaluation:
```bash
python src/evaluation/face_recognition_evaluation.py
```
*Measures Genuine Match Rate (GMR), False Match Rate (FMR), and threshold sweep.*

### Anti-Spoofing & Liveness Evaluation:
```bash
python src/evaluation/liveness_evaluation.py
```
*Measures response against printed photos, screens, replays, and latency.*

### Runtime Performance Benchmark:
```bash
python src/evaluation/performance_benchmark.py
```
*Measures execution latency for MTCNN, InceptionResnetV1, SQLite, and FPS.*

---

## 17. Bus & Route Configuration

All operational constants are centrally managed in [`src/config.py`](file:///d:/SmartBusFaceRecognition/src/config.py):
- `DEFAULT_BUS_ID = "BUS-12"`
- `DEFAULT_ROUTE = "R-101"`
- `DEFAULT_RECOGNITION_THRESHOLD = 0.60`
- `DEFAULT_COOLDOWN_SECONDS = 300` (5 minutes)

---

## 18. Liveness Detection Configuration

Adjust liveness sensitivity in [`src/config.py`](file:///d:/SmartBusFaceRecognition/src/config.py):
- `LIVENESS_ENABLED = True` (Set to `False` to run in development bypass mode)
- `CHALLENGE_TIMEOUT_SECONDS = 3.0` (Seconds passenger has to turn their head)
- `MIN_FRAMES_REQUIRED = 3` (Minimum frames needed before granting entry)
- `TURN_LEFT_YAW_MAX = 0.35` (Yaw threshold for left turn)
- `TURN_RIGHT_YAW_MIN = 0.65` (Yaw threshold for right turn)

---

## 19. Troubleshooting Guide

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **Webcam fails to open (Index 0)** | Camera blocked by Windows privacy settings or used by Zoom/Teams. | Go to Windows Settings $\rightarrow$ Privacy $\rightarrow$ Camera $\rightarrow$ Allow desktop apps. Pass `--camera 1` if using external USB cam. |
| **All faces show UNKNOWN PERSON** | Embeddings have not been generated yet. | Open Streamlit $\rightarrow$ **Face Data** $\rightarrow$ Click **Generate Embeddings**, or run `python -m src.vision.build_student_embeddings`. |
| **Liveness fails immediately** | Passenger is holding too still or not turning head. | Watch the yellow prompt on screen and turn head slightly left or right when instructed. |
| **Duplicate boarding rejection** | Passenger scanned twice within 5 minutes. | This is expected behavior. Wait 5 minutes, or clear test logs in SQLite for testing. |

---

## 20. Prototype Limitations

1. **2D Video Replay Vulnerability:** While static photos and uncoordinated replays are rejected, synchronized interactive video replays could fool 2D RGB cameras. Production systems should add infrared/depth sensors.
2. **Extreme Lighting & Occlusions:** Direct blinding sunlight or dark sunglasses can obscure MTCNN eye landmarks. In such cases, use manual conductor verification.
3. **Single Camera Multi-Face Queuing:** In crowded entryways, passengers should step forward one by one for optimal recognition speed.

---

## 21. Privacy Considerations & Ethics

- **Zero Cloud Transmission:** All deep learning inference executes 100% locally on the device.
- **Biometric Vector Masking:** Raw 512-D float vectors are never displayed in the UI, printed in logs, or exported to CSV files.
- **Ephemeral RAM Video:** Video frames and landmarks exist only in volatile RAM; no video recordings or face snapshots are saved to disk.
- **Mandatory Manual Fallback:** If recognition or liveness fails for any reason, students are never stranded. Conductors inspect physical student ID cards and manually authorize entry.
