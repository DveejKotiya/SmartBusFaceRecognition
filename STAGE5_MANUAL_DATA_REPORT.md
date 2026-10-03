# Stage 5 Report: Manual Student & Bus Pass Data Workflow
## AI-Based Face Recognition and Smart Bus Pass Verification System

---

## 1. Executive Summary

As requested, the project has been updated so that all student data, bus passes, routes, and face photos are **entered and curated manually by the user**, completely bypassing automatic registration.

### New Architecture & Workflow:
1. **Manual CSV Entry**: User records student identity, semester, route, bus ID, and pass validity dates into `data/students/students.csv`.
2. **Database Synchronization (`src/data/student_importer.py`)**: Validates every row and synchronizes with the SQLite database (`students`, `bus_routes`, and `bus_passes` tables) using parameterized SQL queries.
3. **Manual Photo Folders**: User places student photos in `data/students/faces/<student_id>/`.
4. **Offline Embedding Generation (`src/vision/build_student_embeddings.py`)**: Uses MTCNN and pretrained FaceNet to extract embeddings for valid photos, averages them into a centroid profile, and persists them into `models/embeddings.pkl`.
5. **Bus Entrance Verification**: The live terminal and `PassVerifier` seamlessly recognize students and verify their pass using the manually imported data as the sole source of truth.

---

## 2. Files Created & Modified

| File | Purpose | Status |
| :--- | :--- | :---: |
| `data/students/students.csv` | Primary manual CSV spreadsheet with sample student records | **Created** |
| `data/students/students_template.csv` | Clean template reference with 2 clearly marked sample rows | **Created** |
| `src/data/__init__.py` | Data package exports | **Created** |
| `src/data/student_importer.py` | CSV validation, deduplication, and SQLite database synchronizer | **Created** |
| `src/vision/build_student_embeddings.py` | Scans image folders, runs MTCNN/FaceNet, averages embeddings to store | **Created** |
| `tests/test_student_importer.py` | Unit tests for CSV validation, error reporting, and database synchronization | **Created** |
| `tests/test_build_student_embeddings.py` | Unit tests for folder scanning, single-face validation, and embedding generation | **Created** |
| `docs/manual_data_entry.md` | Beginner-friendly 7-step guide for entering and updating student data | **Created** |
| `docs/face_image_requirements.md` | Photo quality, angle, lighting, and privacy guidelines | **Created** |

---

## 3. CSV File Structure

The file [`data/students/students.csv`](file:///d:/SmartBusFaceRecognition/data/students/students.csv) uses 9 columns:

```csv
student_id,name,department,semester,bus_id,route,pass_start,pass_end,pass_status
21CS101,Aarav Sharma,Computer Science,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
21EC202,Priya Patel,Electronics,6,BUS-12,R-101,2026-01-01,2026-08-31,ACTIVE
21ME303,Rohan Gupta,Mechanical,6,BUS-12,R-102,2026-07-01,2026-12-31,ACTIVE
```

### Validation Rules Enforced:
- `student_id`: Non-empty alphanumeric string (Roll Number).
- `name`: Non-empty string.
- `route`: Non-empty string (Route code, e.g. `R-101`).
- `bus_id`: Non-empty string (Bus number, e.g. `BUS-12`).
- `pass_start` & `pass_end`: Valid ISO dates (`YYYY-MM-DD`).
- Date order: `pass_start <= pass_end` required.
- `pass_status`: Must be one of `ACTIVE`, `INACTIVE`, or `EXPIRED`.

---

## 4. Manual Face Image Directory Structure

Face images are organized in folders named after each `student_id`:
```
data/students/faces/
├── 21CS101/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── 21EC202/
│   └── 01.jpg
└── 21ME303/
    ├── 01.jpg
    └── 02.jpg
```

---

## 5. Offline Embedding Generation Workflow

Executed via:
```bash
python -m src.vision.build_student_embeddings
```
For each student directory:
1. **Folder Scan**: Discovers all supported images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. **Face Detection (MTCNN)**:
   - Exactly 1 face detected $\rightarrow$ **ACCEPTED**.
   - 0 faces detected $\rightarrow$ **REJECTED** (no face found).
   - > 1 faces detected $\rightarrow$ **REJECTED** (ambiguous / group photo).
3. **512-D Embedding Extraction (FaceNet)**:
   - Cropped 160x160 face passed through pretrained `InceptionResnetV1`.
4. **Centroid Profile Averaging**:
   - Multiple valid embeddings for a student are averaged:
     $$\mathbf{e}_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{e}_i, \quad \mathbf{e}_{\text{student}} = \frac{\mathbf{e}_{\text{mean}}}{\|\mathbf{e}_{\text{mean}}\|_2}$$
   - Produces an optimal 512-D representation robust to angle and expression variations.
5. **Persistence**: Saved to [`models/embeddings.pkl`](file:///d:/SmartBusFaceRecognition/models/embeddings.pkl).

---

## 6. Update Behavior & Decoupling

- **Metadata Changes**: If a student changes semester, route, bus, or pass dates, simply update `students.csv` and rerun `python -m src.data.student_importer`. Face embeddings do **not** need to be recomputed.
- **Photo Updates**: If you add new or better photos for a student, rerun `python -m src.vision.build_student_embeddings`.

---

## 7. Known Limitations

- Image quality depends on manually supplied files; blurry photos or photos taken in pitch darkness will be rejected by MTCNN.
- Students must have at least one photo with a single detectable face to generate an active embedding.
- In accordance with project instructions, no Streamlit dashboard was created yet; data management is handled entirely via CSV and image folders.
