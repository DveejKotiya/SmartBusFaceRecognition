"""
Real Data Validation & Embedding Generation Pipeline
Smart Bus Face Recognition and Pass Verification System

Executes Parts 1 through 7 of the Real Data Validation stage:
1. Validates data/students/students.csv (structure, types, dates, duplicates).
2. Validates face folder existence for every student.
3. Validates face images (MTCNN detection, single-face verification).
4. Imports student records into SQLite.
5. Generates 512-D face embeddings via FaceNet.
6. Verifies embedding database integrity.
7. Performs self-recognition sanity test.
8. Writes docs/real_data_validation_report.md.
"""

import sys
import csv
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_db_connection, DB_PATH
from src.data.student_importer import StudentImporter, DEFAULT_CSV_PATH, VALID_PASS_STATUSES
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer, compare_embeddings
from src.vision.embedding_store import EmbeddingStore

DEFAULT_FACES_DIR = PROJECT_ROOT / "data" / "students" / "faces"
REPORT_PATH = PROJECT_ROOT / "docs" / "real_data_validation_report.md"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_COLUMNS = [
    "student_id", "name", "department", "semester",
    "bus_id", "route", "pass_start", "pass_end", "pass_status"
]


# =====================================================================
# DATA CONTAINERS FOR VALIDATION RESULTS
# =====================================================================

@dataclass
class CSVValidationReport:
    file_exists: bool = False
    total_raw_rows: int = 0
    valid_student_rows: int = 0
    duplicate_ids: List[str] = field(default_factory=list)
    duplicate_rows: List[int] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    row_errors: List[Dict[str, Any]] = field(default_factory=list)
    students: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FolderValidationReport:
    folders_checked: int = 0
    folders_found: int = 0
    missing_folders: List[str] = field(default_factory=list)
    student_image_counts: Dict[str, int] = field(default_factory=dict)
    students_missing_images: List[str] = field(default_factory=list)


@dataclass
class ImageValidationReport:
    total_images_scanned: int = 0
    valid_single_face_images: int = 0
    no_face_images: List[str] = field(default_factory=list)
    multiple_faces_images: List[str] = field(default_factory=list)
    unreadable_corrupted_images: List[str] = field(default_factory=list)
    student_image_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class SelfRecognitionResult:
    tested: bool = False
    query_image_path: Optional[str] = None
    query_student_id: Optional[str] = None
    predicted_student_id: Optional[str] = None
    predicted_student_name: Optional[str] = None
    top_similarity: float = 0.0
    second_best_similarity: float = 0.0
    second_best_student_id: Optional[str] = None
    match_correct: bool = False
    reason: Optional[str] = None


# =====================================================================
# PART 1: CSV VALIDATION
# =====================================================================

def validate_students_csv(csv_path: Path) -> CSVValidationReport:
    report = CSVValidationReport()
    if not csv_path.exists():
        report.file_exists = False
        return report

    report.file_exists = True
    seen_ids: Set[str] = set()
    seen_rows: Set[str] = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            report.missing_columns = REQUIRED_COLUMNS.copy()
            return report

        # Check required columns
        for col in REQUIRED_COLUMNS:
            if col not in reader.fieldnames:
                report.missing_columns.append(col)

        for idx, row in enumerate(reader, start=2):
            raw_line = ",".join(str(v).strip() for v in row.values())
            if not any(row.values()) or raw_line.startswith("#"):
                continue

            report.total_raw_rows += 1

            # Check duplicate identical row
            if raw_line in seen_rows:
                report.duplicate_rows.append(idx)
                report.row_errors.append({
                    "row_num": idx,
                    "student_id": row.get("student_id", "").strip(),
                    "error": f"Row {idx} is an exact duplicate of an earlier row."
                })
                continue
            seen_rows.add(raw_line)

            # Check individual fields
            sid = row.get("student_id", "").strip().upper()
            name = row.get("name", "").strip()
            dept = row.get("department", "").strip()
            sem = row.get("semester", "").strip()
            bus_id = row.get("bus_id", "").strip().upper()
            route = row.get("route", "").strip().upper()
            p_start = row.get("pass_start", "").strip()
            p_end = row.get("pass_end", "").strip()
            p_status = row.get("pass_status", "").strip().upper()

            errors = []
            if not sid:
                errors.append("'student_id' cannot be empty.")
            elif sid in seen_ids:
                report.duplicate_ids.append(sid)
                errors.append(f"Duplicate student_id '{sid}' encountered.")
            else:
                seen_ids.add(sid)

            if not name:
                errors.append("'name' cannot be empty.")
            if not dept:
                errors.append("'department' cannot be empty.")
            if not bus_id:
                errors.append("'bus_id' cannot be empty.")
            if not route:
                errors.append("'route' cannot be empty.")

            # Validate date formats and ordering
            d_start = None
            d_end = None
            try:
                d_start = datetime.strptime(p_start, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"Invalid pass_start '{p_start}' (expected YYYY-MM-DD).")

            try:
                d_end = datetime.strptime(p_end, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"Invalid pass_end '{p_end}' (expected YYYY-MM-DD).")

            if d_start and d_end and d_start > d_end:
                errors.append(f"pass_start ({d_start}) cannot be after pass_end ({d_end}).")

            if p_status not in VALID_PASS_STATUSES:
                errors.append(f"Invalid pass_status '{p_status}'. Must be one of {sorted(VALID_PASS_STATUSES)}.")

            if errors:
                for err in errors:
                    report.row_errors.append({
                        "row_num": idx,
                        "student_id": sid or f"Row {idx}",
                        "error": err
                    })
            else:
                report.valid_student_rows += 1
                report.students.append({
                    "student_id": sid,
                    "name": name,
                    "department": dept,
                    "semester": sem,
                    "bus_id": bus_id,
                    "route": route,
                    "pass_start": p_start,
                    "pass_end": p_end,
                    "pass_status": p_status
                })

    return report


# =====================================================================
# PART 2: FOLDER VALIDATION
# =====================================================================

def validate_face_folders(students: List[Dict[str, str]], faces_dir: Path) -> FolderValidationReport:
    report = FolderValidationReport()
    report.folders_checked = len(students)

    for s in students:
        sid = s["student_id"]
        folder = faces_dir / sid

        if not folder.exists() or not folder.is_dir():
            report.missing_folders.append(sid)
            report.student_image_counts[sid] = 0
            report.students_missing_images.append(sid)
        else:
            report.folders_found += 1
            images = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
            count = len(images)
            report.student_image_counts[sid] = count
            if count == 0:
                report.students_missing_images.append(sid)

    return report


# =====================================================================
# PART 3: IMAGE VALIDATION
# =====================================================================

def validate_face_images(
    students: List[Dict[str, str]],
    faces_dir: Path,
    detector: FaceDetector
) -> ImageValidationReport:
    report = ImageValidationReport()

    for s in students:
        sid = s["student_id"]
        folder = faces_dir / sid
        stats = {
            "total": 0,
            "valid": 0,
            "no_face": 0,
            "multiple_faces": 0,
            "unreadable": 0
        }

        if folder.exists() and folder.is_dir():
            image_files = sorted([
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ])
            stats["total"] = len(image_files)
            report.total_images_scanned += len(image_files)

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["unreadable"] += 1
                    report.unreadable_corrupted_images.append(str(img_path))
                    continue

                try:
                    detections = detector.detect_faces(img, is_bgr=True)
                    if len(detections) == 1:
                        stats["valid"] += 1
                        report.valid_single_face_images += 1
                    elif len(detections) == 0:
                        stats["no_face"] += 1
                        report.no_face_images.append(str(img_path))
                    else:
                        stats["multiple_faces"] += 1
                        report.multiple_faces_images.append(str(img_path))
                except Exception as e:
                    stats["unreadable"] += 1
                    report.unreadable_corrupted_images.append(f"{img_path} ({e})")

        report.student_image_stats[sid] = stats

    return report


# =====================================================================
# PART 7: TEST SELF-RECOGNITION
# =====================================================================

def test_self_recognition(
    students: List[Dict[str, str]],
    faces_dir: Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    store: EmbeddingStore
) -> SelfRecognitionResult:
    result = SelfRecognitionResult()

    if len(store) == 0:
        result.tested = False
        result.reason = "Embedding store is empty (no registered embeddings)."
        return result

    # Find first student with at least one valid image
    target_img_path = None
    target_student_id = None

    for s in students:
        sid = s["student_id"]
        folder = faces_dir / sid
        if folder.exists() and folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    img = cv2.imread(str(f))
                    if img is not None:
                        dets = detector.detect_faces(img, is_bgr=True)
                        if len(dets) == 1:
                            target_img_path = f
                            target_student_id = sid
                            break
        if target_img_path:
            break

    if not target_img_path:
        result.tested = False
        result.reason = "No valid single-face images found in any student folder."
        return result

    result.tested = True
    result.query_image_path = str(target_img_path)
    result.query_student_id = target_student_id

    # Generate embedding for query image
    img = cv2.imread(str(target_img_path))
    dets = detector.detect_faces(img, is_bgr=True)
    query_emb = recognizer.generate_embedding(dets[0].face_crop)

    # Compare against all stored embeddings
    all_stored = store.get_all()
    scores = []
    for rec in all_stored:
        sim = compare_embeddings(query_emb, rec["embedding"])
        scores.append((sim, rec["student_id"], rec["student_name"]))

    # Sort descending
    scores.sort(key=lambda x: x[0], reverse=True)

    if scores:
        best_sim, best_id, best_name = scores[0]
        result.predicted_student_id = str(best_id)
        result.predicted_student_name = best_name
        result.top_similarity = best_sim
        result.match_correct = (str(best_id).upper() == target_student_id.upper())

        if len(scores) > 1:
            sec_sim, sec_id, _ = scores[1]
            result.second_best_similarity = sec_sim
            result.second_best_student_id = str(sec_id)
        else:
            result.second_best_similarity = 0.0
            result.second_best_student_id = "N/A (only 1 student registered)"

    return result


# =====================================================================
# REPORT WRITER
# =====================================================================

def write_markdown_report(
    csv_rep: CSVValidationReport,
    folder_rep: FolderValidationReport,
    img_rep: ImageValidationReport,
    import_summary: Any,
    store: EmbeddingStore,
    self_rec: SelfRecognitionResult
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Real Data Validation & Embedding Generation Report\n")
        f.write("## Smart Bus Face Recognition and Pass Verification System\n\n")
        f.write(f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 1. CSV Validation
        f.write("## 1. CSV Data Validation (`data/students/students.csv`)\n\n")
        f.write(f"- **File Exists**: {'Yes' if csv_rep.file_exists else 'NO (File Missing)'}\n")
        f.write(f"- **Total Rows Evaluated**: {csv_rep.total_raw_rows}\n")
        f.write(f"- **Valid Student Records**: {csv_rep.valid_student_rows}\n")
        f.write(f"- **Duplicate Student IDs**: {len(csv_rep.duplicate_ids)}\n")
        f.write(f"- **Duplicate Identical Rows**: {len(csv_rep.duplicate_rows)}\n\n")

        if csv_rep.missing_columns:
            f.write(f"**Missing Required Columns**: {', '.join(csv_rep.missing_columns)}\n\n")

        if csv_rep.row_errors:
            f.write("### Row Validation Errors Found:\n")
            for err in csv_rep.row_errors:
                f.write(f"- **Row {err['row_num']}** ({err['student_id']}): {err['error']}\n")
            f.write("\n")
        else:
            f.write("> [!NOTE] All CSV records passed format, date ordering, and status validation.\n\n")

        # 2. Student Folders
        f.write("## 2. Face Folders & Images Status (`data/students/faces/`)\n\n")
        f.write(f"- **Total Students Checked**: {folder_rep.folders_checked}\n")
        f.write(f"- **Folders Found**: {folder_rep.folders_found}\n")
        f.write(f"- **Missing Folders**: {len(folder_rep.missing_folders)}\n\n")

        f.write("| Student ID | Name | Folder Exists? | Images Found | Valid Faces | Rejected Photos |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for s in csv_rep.students:
            sid = s["student_id"]
            name = s["name"]
            has_folder = "✅ Yes" if sid not in folder_rep.missing_folders else "❌ Missing"
            img_count = folder_rep.student_image_counts.get(sid, 0)
            stats = img_rep.student_image_stats.get(sid, {"valid": 0, "no_face": 0, "multiple_faces": 0, "unreadable": 0})
            rej_count = stats["no_face"] + stats["multiple_faces"] + stats["unreadable"]
            f.write(f"| `{sid}` | {name} | {has_folder} | {img_count} | {stats['valid']} | {rej_count} |\n")
        f.write("\n")

        if folder_rep.missing_folders:
            f.write("> [!WARNING] The following student folders do not exist yet in `data/students/faces/`:\n")
            for m in folder_rep.missing_folders:
                f.write(f"> - `data/students/faces/{m}/`\n")
            f.write("\n")

        # 3. Image Processing
        f.write("## 3. MTCNN Face Image Quality Analysis\n\n")
        f.write(f"- **Total Photos Scanned**: {img_rep.total_images_scanned}\n")
        f.write(f"- **Valid Single-Face Photos**: {img_rep.valid_single_face_images}\n")
        f.write(f"- **Photos with Zero Faces**: {len(img_rep.no_face_images)}\n")
        f.write(f"- **Photos with Multiple Faces**: {len(img_rep.multiple_faces_images)}\n")
        f.write(f"- **Unreadable / Corrupted Photos**: {len(img_rep.unreadable_corrupted_images)}\n\n")

        # 4. Database Sync
        f.write("## 4. SQLite Database Synchronization\n\n")
        if import_summary:
            f.write(f"- **New Students Imported**: {import_summary.imported_count}\n")
            f.write(f"- **Existing Students Updated**: {import_summary.updated_count}\n")
            f.write(f"- **Rows Rejected**: {import_summary.rejected_count}\n\n")

        # 5. Embeddings Verification
        f.write("## 5. Biometric Embedding Database Verification (`models/embeddings.pkl`)\n\n")
        f.write(f"- **Total Students in Store**: {len(store)}\n")
        all_recs = store.get_all()
        if all_recs:
            f.write("| Stored Student ID | Name | Embedding Shape | Finite? | Unit L2-Norm |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: |\n")
            for rec in all_recs:
                emb = rec["embedding"]
                is_finite = bool(np.all(np.isfinite(emb)))
                norm = float(np.linalg.norm(emb))
                f.write(f"| `{rec['student_id']}` | {rec['student_name']} | {emb.shape} | {'✅ Yes' if is_finite else '❌ No'} | {norm:.4f} |\n")
            f.write("\n")
        else:
            f.write("> [!NOTICE] The embedding store currently contains 0 records because no face images were available to process.\n\n")

        # 6. Self Recognition Test
        f.write("## 6. Self-Recognition Sanity Test\n\n")
        if self_rec.tested:
            f.write(f"- **Query Image**: `{self_rec.query_image_path}`\n")
            f.write(f"- **Actual Student ID**: `{self_rec.query_student_id}`\n")
            f.write(f"- **Predicted Student ID**: `{self_rec.predicted_student_id}` ({self_rec.predicted_student_name})\n")
            f.write(f"- **Top Match Similarity**: `{self_rec.top_similarity:.4f}`\n")
            f.write(f"- **Second-Best Similarity**: `{self_rec.second_best_similarity:.4f}` ({self_rec.second_best_student_id})\n")
            f.write(f"- **Verification Outcome**: {'✅ Match Confirmed' if self_rec.match_correct else '❌ Mismatch'}\n\n")
            f.write("> [!NOTE] This is a single-image sanity test to verify embedding vector retrieval fidelity. It does not measure generalized statistical accuracy.\n\n")
        else:
            f.write(f"> [!NOTICE] Self-recognition test skipped: {self_rec.reason}\n\n")

        # 7. Next Actions
        f.write("## 7. Recommended Next Actions\n\n")
        if folder_rep.missing_folders or folder_rep.students_missing_images:
            f.write("1. Place student face photos into their designated folders:\n")
            for s in csv_rep.students:
                f.write(f"   - `data/students/faces/{s['student_id']}/01.jpg`, `02.jpg`, etc.\n")
            f.write("2. Rerun `python -m src.vision.build_student_embeddings` to extract embeddings.\n")
            f.write("3. Rerun `python -m src.data.validate_real_data` to re-verify the full pipeline.\n")
        else:
            f.write("All student data, folders, images, and embeddings are 100% synchronized and verified!\n")


# =====================================================================
# MAIN PIPELINE RUNNER
# =====================================================================

def run_real_data_pipeline():
    print("=" * 68)
    print("  REAL DATA VALIDATION & EMBEDDING GENERATION PIPELINE")
    print("=" * 68)

    # PART 1: CSV Validation
    print("\n[PART 1] Validating data/students/students.csv...")
    csv_report = validate_students_csv(DEFAULT_CSV_PATH)
    print(f"  - File Exists              : {csv_report.file_exists}")
    print(f"  - Total Raw Rows           : {csv_report.total_raw_rows}")
    print(f"  - Valid Student Records    : {csv_report.valid_student_rows}")
    print(f"  - Duplicate Student IDs    : {len(csv_report.duplicate_ids)}")

    if csv_report.row_errors:
        print(f"  [WARNING] {len(csv_report.row_errors)} row error(s) found:")
        for err in csv_report.row_errors:
            print(f"    * Row {err['row_num']}: {err['error']}")
    else:
        print("  [SUCCESS] All CSV records passed format, date, and status checks.")

    # PART 2: Face Folders Validation
    print("\n[PART 2] Validating face directories in data/students/faces/...")
    DEFAULT_FACES_DIR.mkdir(parents=True, exist_ok=True)
    folder_report = validate_face_folders(csv_report.students, DEFAULT_FACES_DIR)
    print(f"  - Students Checked         : {folder_report.folders_checked}")
    print(f"  - Folders Found            : {folder_report.folders_found}")
    print(f"  - Missing Folders          : {len(folder_report.missing_folders)}")

    for sid in folder_report.missing_folders:
        print(f"    * Missing folder: data/students/faces/{sid}/")

    for sid, count in folder_report.student_image_counts.items():
        print(f"    * Student {sid:10s} : {count} image(s) found")

    # Initialize vision tools
    print("\n[PART 3] Initializing FaceDetector & FaceRecognizer...")
    detector = FaceDetector(device="cpu")
    recognizer = FaceRecognizer(device="cpu")
    store = EmbeddingStore()

    # Image Validation
    print("\n[PART 3] Validating face image quality with MTCNN...")
    img_report = validate_face_images(csv_report.students, DEFAULT_FACES_DIR, detector)
    print(f"  - Total Images Scanned     : {img_report.total_images_scanned}")
    print(f"  - Valid Single-Face Images : {img_report.valid_single_face_images}")
    print(f"  - Zero Faces Detected      : {len(img_report.no_face_images)}")
    print(f"  - Multiple Faces Detected  : {len(img_report.multiple_faces_images)}")
    print(f"  - Unreadable Images        : {len(img_report.unreadable_corrupted_images)}")

    # PART 4: Import Student Data into SQLite
    print("\n[PART 4] Importing validated student records into SQLite database...")
    importer = StudentImporter()
    import_summary = importer.import_csv(DEFAULT_CSV_PATH)
    print(f"  - Database Synchronized    : {DB_PATH.name}")
    print(f"  - New Students Added       : {import_summary.imported_count}")
    print(f"  - Existing Students Updated: {import_summary.updated_count}")
    print(f"  - Rows Rejected            : {import_summary.rejected_count}")

    # PART 5: Generate Real Face Embeddings
    print("\n[PART 5] Generating real face embeddings...")
    from src.vision.build_student_embeddings import StudentEmbeddingBuilder
    builder = StudentEmbeddingBuilder(faces_dir=DEFAULT_FACES_DIR, device="cpu")
    builder_reports = builder.build_all(csv_report.students)
    store = builder.store

    # PART 6: Verify Embedding Database
    print("\n[PART 6] Verifying biometric embedding store...")
    print(f"  - Total Profiles in Store  : {len(store)}")
    for rec in store.get_all():
        emb = rec["embedding"]
        finite = bool(np.all(np.isfinite(emb)))
        print(f"    * Student {rec['student_id']:10s} ({rec['student_name']}): Shape={emb.shape}, Finite={finite}")

    # PART 7: Test Self-Recognition
    print("\n[PART 7] Testing self-recognition sanity check...")
    self_rec = test_self_recognition(csv_report.students, DEFAULT_FACES_DIR, detector, recognizer, store)
    if self_rec.tested:
        print(f"  - Query Image              : {Path(self_rec.query_image_path).name}")
        print(f"  - Ground Truth ID          : {self_rec.query_student_id}")
        print(f"  - Predicted ID             : {self_rec.predicted_student_id} ({self_rec.predicted_student_name})")
        print(f"  - Match Similarity Score   : {self_rec.top_similarity:.4f}")
        print(f"  - Second-Best Score        : {self_rec.second_best_similarity:.4f} ({self_rec.second_best_student_id})")
        print(f"  - Result                   : {'SUCCESS' if self_rec.match_correct else 'MISMATCH'}")
    else:
        print(f"  - Status                   : Skipped ({self_rec.reason})")

    # PART 8: Write Report
    print("\n[PART 8] Writing docs/real_data_validation_report.md...")
    write_markdown_report(csv_report, folder_report, img_report, import_summary, store, self_rec)
    print(f"  - Saved report to: {REPORT_PATH}")

    print("\n" + "=" * 68)
    print("  REAL DATA VALIDATION PIPELINE FINISHED")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    run_real_data_pipeline()
