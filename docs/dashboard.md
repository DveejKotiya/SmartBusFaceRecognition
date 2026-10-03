# Streamlit Admin Dashboard User Guide & Technical Documentation

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 8 — Admin Dashboard  
**Status:** Implemented & Verified  

---

## 1. Overview

The **Smart Bus Admin Dashboard** is a web-based operational and reporting interface built with [Streamlit](https://streamlit.io/). It provides bus coordinators, transport administrators, and campus operators with full oversight of:

- Student registrations and bus pass allocations
- Biometric template enrollment status
- Real-time live camera bus entrance verification terminal
- Immutable entrance audit logs
- Multi-criteria reporting and CSV exports
- System health and hardware diagnostics

---

## 2. Launching the Dashboard

### 2.1 Prerequisites
Ensure the virtual environment with dependencies is activated. The following packages are required (installed via `requirements.txt`):
- `streamlit`
- `torch`, `torchvision`, `facenet-pytorch`
- `opencv-python`
- `pandas`
- `scikit-learn`

### 2.2 Startup Command
From the project root directory (`D:\SmartBusFaceRecognition`), run:

```bash
streamlit run app.py
```

Streamlit will launch a local web server (typically at `http://localhost:8501`) and automatically open the application in your default browser.

> [!NOTE]
> The webcam will **not** start automatically when the application opens. The camera only activates when an operator navigates to the **Live Bus Entry** page and clicks **Start Live Camera**.

---

## 3. Architecture & Modular Navigation

The application uses a modular multi-page architecture located in `src/ui/`:

```
d:/SmartBusFaceRecognition/
├── app.py                      # Main entry point, navigation router, privacy banner
└── src/
    ├── ui/
    │   ├── __init__.py
    │   ├── dashboard_page.py   # Page 1: Operations summary & recent boardings
    │   ├── students_page.py    # Page 2: Student directory & search
    │   ├── passes_page.py      # Page 3: Bus pass directory & expiry warnings
    │   ├── face_data_page.py   # Page 4: Biometric status & embedding builder
    │   ├── live_entry_page.py  # Page 5: Live camera entrance verification terminal
    │   ├── logs_page.py        # Page 6: Boarding audit log viewer
    │   ├── reports_page.py     # Page 7: Filtered reporting & CSV exports
    │   └── system_page.py      # Page 8: Environment, DB, & hardware diagnostics
    ├── database/
    │   └── dashboard_queries.py # Centralized SQL queries (parameterized)
    └── core/
        ├── bus_entry_service.py # Orchestrates vision + verification pipeline
        └── pass_verifier.py    # Rules engine & cooldown enforcement
```

---

## 4. Detailed Page Guide

### 4.1 📊 Dashboard (Home)
- **Purpose**: Provides high-level operational situational awareness at a glance.
- **Metric Cards**:
  - `Registered Students`: Total student records in the SQLite database.
  - `Active Bus Passes`: Current active passes valid as of today.
  - `Expired Passes`: Passes past their expiration date or marked expired.
  - `Today's Boardings (Allowed)`: Successful passenger entries today.
  - `Today's Denied Attempts`: Boarding rejections logged today.
  - `Today's Unknown Faces`: Unrecognized face attempts logged today.
- **Recent Entrance Activity Feed**: Table showing the last 10 entries across all buses, highlighted green for allowed and red for denied.
- **Data Source**: Live queries from `students`, `bus_passes`, and `entry_logs` tables via `src/database/dashboard_queries.py`.

---

### 4.2 🎓 Students Directory
- **Purpose**: Browse and verify enrolled students.
- **Capabilities**:
  - Search by Student ID (e.g., `21CS101`) or Full Name (case-insensitive substring match).
  - Filter by Department (e.g., `Computer Science`, `Electronics`, `Mechanical`).
  - Filter by Semester (e.g., `4`, `6`).
- **Data Lineage**: Synchronized from `data/students/students.csv` into SQLite table `students`.
- **Manual Curation Workflow**: The UI displays a guidance banner reminding operators that `data/students/students.csv` is the manual source of truth. After modifying the CSV, run `python -m src.data.student_importer`.

---

### 4.3 🎫 Bus Passes Directory
- **Purpose**: Manage and monitor student transit authorizations.
- **Capabilities**:
  - Filter by status radio buttons: `ALL`, `ACTIVE`, `EXPIRED`, `INACTIVE`.
  - Color-coded rows: Soft green for active passes, amber/yellow for expired passes.
  - Warning banner alerting operators if any displayed passes have expired.
- **Data Lineage**: Queries `bus_passes` joined with `students` and `bus_routes`.

---

### 4.4 👤 Face Data & Biometric Management
- **Purpose**: Track facial photo enrollment and embedding readiness without exposing sensitive biometrics.
- **Capabilities**:
  - Lists each student with photo count, embedding count, and status:
    - `READY`: Photos present in `data/students/faces/<id>/` and embedding exists in `models/embeddings.pkl`.
    - `MISSING_IMAGES`: No enrollment photos found in the folder.
    - `MISSING_EMBEDDINGS`: Photos exist but embedding vector has not been computed yet.
    - `ERROR`: Inconsistent or corrupt state.
  - Summary metric cards showing count of ready students vs. pending photos/embeddings.
  - One-click action button: **"Process Photos & Build Face Embeddings"**, which triggers `StudentEmbeddingBuilder` directly from the dashboard.
- **Privacy Protections**: Raw 512-dimensional vector arrays are strictly hidden and impossible to download through the UI.

---

### 4.5 📹 Live Bus Entry Terminal
- **Purpose**: Interactive, live bus-door verification station for real passengers.
- **Configuration Controls**:
  - Select active `Bus ID` (e.g., `BUS-12`, `BUS-08`).
  - Select active `Route Code` (e.g., `R-101`, `R-102`).
  - Adjust `Recognition Threshold` slider (default `0.60`).
- **Camera Stream**:
  - **Start Live Camera** / **Stop Camera** buttons. Camera is completely off until started.
  - Left panel: Real-time OpenCV video stream with bounding boxes and passenger annotations.
  - Right panel: Live passenger decision card showing:
    - 🟢 `ENTRY ALLOWED`: Passenger name, roll number, match similarity, route verified.
    - 🟡 `UNKNOWN PERSON`: Unrecognized face prompt to show physical college ID.
    - 🔴 `ENTRY DENIED`: Specific refusal reason (e.g., expired pass, wrong route, cooldown active).
- **Service Integration**: Calls the cached singleton `BusEntryService`, running MTCNN detection and InceptionResnetV1 embedding matching.

---

### 4.6 📝 Entry Logs Audit Page
- **Purpose**: Complete historical audit trail of all entry events.
- **Filters**:
  - Date picker (optional checkbox).
  - Passenger search (student ID or name substring).
  - Bus ID dropdown.
  - Route Code dropdown.
  - Decision dropdown (`ALL`, `APPROVED`, `REJECTED`, `UNKNOWN`).
  - Rows to display slider (25, 50, 100, 250).
- **Data Source**: Parameterized SQL queries against SQLite `entry_logs`.

---

### 4.7 📈 Reports & Data Export
- **Purpose**: Executive summaries and official CSV exports for campus transportation planning.
- **Report Types**:
  1. `Daily Entrance Report`: Single-day boarding logs and summary cards.
  2. `Date-Range Audit Report`: Multi-day transit summary.
  3. `Route-Specific Boarding Report`: Volume analysis per bus route.
  4. `Student Attendance History`: Individual student transit logs.
- **CSV Export**:
  - Generates downloadable CSV matching official schema:
    `entry_id, timestamp, student_id, student_name, bus_id, route, similarity, result, reason`
  - Biometric vectors and sensitive private tokens are strictly omitted from the export.

---

### 4.8 ⚙️ System Status & Diagnostics
- **Purpose**: Self-test diagnostic panel verifying runtime health.
- **System Checks**:
  - Core Runtimes: Python version, PyTorch version, OpenCV version, Streamlit version, CUDA GPU status.
  - Storage & Integrity: SQLite database connectivity, initialized table count, `models/embeddings.pkl` existence, enrolled biometric count.
  - Hardware Check: "Test Webcam Access" button pings default video device (index 0) and reports accessibility.

---

## 5. Security & Privacy Highlights

1. **Zero Cloud Dependencies**: 100% of recognition and verification occurs on-premises.
2. **Biometric Privacy**: Raw embedding vectors are never rendered or exported.
3. **5-Minute Anti-Passback Cooldown**: Prevents rapid duplicate boarding attempts using database timestamps.
4. **Parameterized Queries**: All database lookups in `dashboard_queries.py` utilize parameterized SQL queries to prevent SQL injection.
5. **Manual Fallback**: Students unrecognized by computer vision can always be verified manually via physical ID cards and the student search interface.

---

## 6. Known Prototype Limitations

- **Camera Hardware Concurrency**: If another software (or a terminal script like `src/bus_entry_camera.py`) is using webcam index 0, the Streamlit live camera will report a capture device busy error.
- **Single-Node SQLite**: SQLite is optimal for local college bus nodes; multi-bus cloud synchronization will be addressed in future stages.
- **No Anti-Spoofing (Stage 8)**: The system assumes cooperative passengers presenting their real faces. Liveness and anti-spoofing detection will be implemented in subsequent stages.
