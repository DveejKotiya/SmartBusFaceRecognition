"""
Student Face Embeddings Builder
Smart Bus Face Recognition and Pass Verification System

This module scans manually supplied face images organized by student ID:
    data/students/faces/<student_id>/*.jpg

For each student:
1. Reads all face images in their folder.
2. Detects faces using the existing FaceDetector (MTCNN).
3. Rejects images that contain 0 faces or more than 1 face.
4. Extracts 512-dimensional embeddings for all valid faces using FaceRecognizer (FaceNet).
5. Combines multiple valid embeddings into an optimal unit-normalized centroid vector.
6. Persists the student embeddings into EmbeddingStore (models/embeddings.pkl).

IMPORTANT:
Uses strictly pretrained inference models. Does NOT train or fine-tune any network.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.embedding_store import EmbeddingStore
from src.database.db_connection import get_db_connection, DB_PATH
from src.data.student_importer import DEFAULT_CSV_PATH

DEFAULT_FACES_DIR = PROJECT_ROOT / "data" / "students" / "faces"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_students_to_process(db_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Retrieves the list of active students to process from the database.
    Falls back to reading students.csv if the database has not been initialized.
    """
    target_db = db_path if db_path else DB_PATH
    students = []

    if target_db.exists():
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT roll_number, name FROM students WHERE is_active = 1 ORDER BY roll_number")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                return [{"student_id": r["roll_number"], "name": r["name"]} for r in rows]
        except Exception as e:
            print(f"[NOTICE] Could not query database ({e}); falling back to CSV.")

    # Fallback to reading students.csv
    if DEFAULT_CSV_PATH.exists():
        import csv
        with open(DEFAULT_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "").strip()
                name = row.get("name", "").strip()
                if sid and not sid.startswith("#"):
                    students.append({"student_id": sid.upper(), "name": name})

    return students


class StudentEmbeddingBuilder:
    """
    Scans student image folders and builds persistent facial embeddings.
    """

    def __init__(
        self,
        faces_dir: Optional[Union[str, Path]] = None,
        store_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initializes detector, recognizer, and embedding store.
        """
        self.faces_dir = Path(faces_dir) if faces_dir else DEFAULT_FACES_DIR
        self.detector = FaceDetector(device=device)
        self.recognizer = FaceRecognizer(device=device)
        self.store = EmbeddingStore(storage_path=store_path)

    def process_student_folder(
        self,
        student_id: str,
        student_name: str
    ) -> Dict[str, Any]:
        """
        Processes all images in data/students/faces/<student_id>/.

        Rules:
        - Exactly 1 face detected -> ACCEPTED
        - 0 faces detected -> REJECTED (no face found)
        - > 1 faces detected -> REJECTED (ambiguous identity / crowd photo)
        """
        # Security: sanitize student_id to prevent directory traversal
        clean_id = Path(str(student_id).strip()).name
        folder = (self.faces_dir / clean_id).resolve()
        try:
            if not folder.is_relative_to(self.faces_dir.resolve()):
                result["status"] = "INVALID_PATH"
                return result
        except AttributeError:
            pass

        result = {
            "student_id": clean_id,
            "student_name": student_name,
            "images_found": 0,
            "valid_images": 0,
            "rejected_images": 0,
            "embeddings_generated": 0,
            "status": "NO_FOLDER"
        }

        if not folder.exists() or not folder.is_dir():
            result["status"] = "FOLDER_NOT_FOUND"
            return result

        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        result["images_found"] = len(image_files)

        if not image_files:
            result["status"] = "NO_IMAGES"
            return result

        valid_embeddings: List[np.ndarray] = []

        for img_path in sorted(image_files):
            # Load image using OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                result["rejected_images"] += 1
                continue

            try:
                # Detect faces with MTCNN
                detections = self.detector.detect_faces(img, is_bgr=True)

                if len(detections) == 1:
                    # Exactly one face found: extract 512-D embedding
                    emb = self.recognizer.generate_embedding(detections[0].face_crop)
                    valid_embeddings.append(emb)
                    result["valid_images"] += 1
                else:
                    # Either 0 faces or multiple faces
                    result["rejected_images"] += 1
            except Exception:
                result["rejected_images"] += 1

        result["embeddings_generated"] = len(valid_embeddings)

        if valid_embeddings:
            # Average the embeddings to create a robust centroid profile
            mean_vector = np.mean(valid_embeddings, axis=0)
            norm = np.linalg.norm(mean_vector)
            centroid = mean_vector / (norm + 1e-10)

            # Store in the persistent EmbeddingStore
            self.store.add_embedding(
                student_id=student_id,
                student_name=student_name,
                embedding=centroid
            )
            result["status"] = "SUCCESS"
        else:
            result["status"] = "NO_VALID_FACES"

        return result

    def build_all(
        self,
        students: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds embeddings for all provided or discovered students.
        """
        if students is None:
            students = get_students_to_process()

        reports = []
        for s in students:
            rep = self.process_student_folder(s["student_id"], s["name"])
            reports.append(rep)

        # Save store to disk if any embeddings were generated
        self.store.save()
        return reports


def run_cli():
    """Command-line execution."""
    print("=" * 65)
    print("  SMART BUS FACE EMBEDDINGS BUILDER")
    print("=" * 65)
    print(f"Scanning face images folder: {DEFAULT_FACES_DIR}")

    students = get_students_to_process()
    if not students:
        print("[WARNING] No students found in database or CSV.")
        print("          Please run: python -m src.data.student_importer")
        return

    print(f"Found {len(students)} student(s) to process.\n")
    builder = StudentEmbeddingBuilder()
    reports = builder.build_all(students)

    print("-" * 65)
    print("INDIVIDUAL STUDENT PROCESSING REPORTS:")
    print("-" * 65)

    total_valid = 0
    total_stored = 0

    for rep in reports:
        sid = rep["student_id"]
        name = rep["student_name"]
        print(f"\nStudent             : {sid} ({name})")
        print(f"Images found        : {rep['images_found']}")
        print(f"Valid images        : {rep['valid_images']}")
        print(f"Rejected images     : {rep['rejected_images']}")
        print(f"Embeddings generated: {rep['embeddings_generated']}")

        if rep["status"] == "SUCCESS":
            print(f"Result              : [OK] Stored in biometric database.")
            total_stored += 1
        elif rep["status"] == "FOLDER_NOT_FOUND":
            print(f"Result              : [NOTICE] Folder data/students/faces/{sid}/ does not exist yet.")
        elif rep["status"] == "NO_IMAGES":
            print(f"Result              : [NOTICE] No .jpg/.png images found in data/students/faces/{sid}/.")
        elif rep["status"] == "NO_VALID_FACES":
            print(f"Result              : [WARNING] All images were rejected (no face or >1 face detected).")

        total_valid += rep["valid_images"]

    print("\n" + "=" * 65)
    print(f"BUILD SUMMARY:")
    print(f"  Students with active embeddings : {total_stored} of {len(students)}")
    print(f"  Total valid photos processed    : {total_valid}")
    print(f"  Embeddings store saved to       : {builder.store.storage_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_cli()
