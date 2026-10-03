# Real Data Validation & Embedding Generation Report
## Smart Bus Face Recognition and Pass Verification System

**Generated At**: 2026-10-03 12:42:26

---

## 1. CSV Data Validation (`data/students/students.csv`)

- **File Exists**: Yes
- **Total Rows Evaluated**: 3
- **Valid Student Records**: 3
- **Duplicate Student IDs**: 0
- **Duplicate Identical Rows**: 0

> [!NOTE] All CSV records passed format, date ordering, and status validation.

## 2. Face Folders & Images Status (`data/students/faces/`)

- **Total Students Checked**: 3
- **Folders Found**: 0
- **Missing Folders**: 3

| Student ID | Name | Folder Exists? | Images Found | Valid Faces | Rejected Photos |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `21CS101` | Aarav Sharma | ❌ Missing | 0 | 0 | 0 |
| `21EC202` | Priya Patel | ❌ Missing | 0 | 0 | 0 |
| `21ME303` | Rohan Gupta | ❌ Missing | 0 | 0 | 0 |

> [!WARNING] The following student folders do not exist yet in `data/students/faces/`:
> - `data/students/faces/21CS101/`
> - `data/students/faces/21EC202/`
> - `data/students/faces/21ME303/`

## 3. MTCNN Face Image Quality Analysis

- **Total Photos Scanned**: 0
- **Valid Single-Face Photos**: 0
- **Photos with Zero Faces**: 0
- **Photos with Multiple Faces**: 0
- **Unreadable / Corrupted Photos**: 0

## 4. SQLite Database Synchronization

- **New Students Imported**: 0
- **Existing Students Updated**: 3
- **Rows Rejected**: 0

## 5. Biometric Embedding Database Verification (`models/embeddings.pkl`)

- **Total Students in Store**: 0
> [!NOTICE] The embedding store currently contains 0 records because no face images were available to process.

## 6. Self-Recognition Sanity Test

> [!NOTICE] Self-recognition test skipped: Embedding store is empty (no registered embeddings).

## 7. Recommended Next Actions

1. Place student face photos into their designated folders:
   - `data/students/faces/21CS101/01.jpg`, `02.jpg`, etc.
   - `data/students/faces/21EC202/01.jpg`, `02.jpg`, etc.
   - `data/students/faces/21ME303/01.jpg`, `02.jpg`, etc.
2. Rerun `python -m src.vision.build_student_embeddings` to extract embeddings.
3. Rerun `python -m src.data.validate_real_data` to re-verify the full pipeline.
