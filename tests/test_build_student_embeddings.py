"""
Unit Tests for StudentEmbeddingBuilder
Smart Bus Face Recognition and Pass Verification System
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import tempfile
from unittest.mock import MagicMock
import cv2
import numpy as np

from src.vision.build_student_embeddings import StudentEmbeddingBuilder
from src.vision.face_detector import DetectionResult
from src.vision.embedding_store import EmbeddingStore


class TestStudentEmbeddingBuilder(unittest.TestCase):
    """Deterministic test suite for building student embeddings from image folders."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.faces_dir = Path(self.temp_dir.name) / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = Path(self.temp_dir.name) / "test_store.pkl"

        # Initialize builder pointing to temporary directories
        self.builder = StudentEmbeddingBuilder(
            faces_dir=self.faces_dir,
            store_path=self.store_path,
            device="cpu"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_image(self, folder: Path, filename: str) -> Path:
        """Helper to create a dummy image on disk."""
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / filename
        # 160x160 blank image
        blank = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(file_path), blank)
        return file_path

    def test_missing_face_image_directory(self):
        """Verify handling of missing student directory."""
        rep = self.builder.process_student_folder("NONEXISTENT", "Nobody")
        self.assertEqual(rep["status"], "FOLDER_NOT_FOUND")
        self.assertEqual(rep["images_found"], 0)

    def test_empty_face_image_directory(self):
        """Verify handling of empty student directory."""
        student_dir = self.faces_dir / "21CS101"
        student_dir.mkdir(parents=True, exist_ok=True)

        rep = self.builder.process_student_folder("21CS101", "Aarav Sharma")
        self.assertEqual(rep["status"], "NO_IMAGES")
        self.assertEqual(rep["images_found"], 0)

    def test_image_with_no_face_rejected(self):
        """Verify images where MTCNN detects 0 faces are rejected."""
        student_dir = self.faces_dir / "21CS101"
        self._create_dummy_image(student_dir, "01.jpg")

        # Mock detector to return 0 detections (no face)
        self.builder.detector.detect_faces = MagicMock(return_value=[])

        rep = self.builder.process_student_folder("21CS101", "Aarav Sharma")
        self.assertEqual(rep["images_found"], 1)
        self.assertEqual(rep["valid_images"], 0)
        self.assertEqual(rep["rejected_images"], 1)
        self.assertEqual(rep["embeddings_generated"], 0)
        self.assertEqual(rep["status"], "NO_VALID_FACES")

    def test_image_with_multiple_faces_rejected(self):
        """Verify images where MTCNN detects >1 faces are rejected."""
        student_dir = self.faces_dir / "21CS101"
        self._create_dummy_image(student_dir, "group_photo.jpg")

        # Mock detector to return 2 detections
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        det1 = DetectionResult(box=(10, 10, 50, 50), confidence=0.95, face_crop=dummy_crop)
        det2 = DetectionResult(box=(60, 60, 100, 100), confidence=0.92, face_crop=dummy_crop)
        self.builder.detector.detect_faces = MagicMock(return_value=[det1, det2])

        rep = self.builder.process_student_folder("21CS101", "Aarav Sharma")
        self.assertEqual(rep["images_found"], 1)
        self.assertEqual(rep["valid_images"], 0)
        self.assertEqual(rep["rejected_images"], 1)

    def test_valid_face_image_and_embedding_generation(self):
        """Verify valid images produce 512-D embeddings and save to store."""
        student_dir = self.faces_dir / "21CS101"
        self._create_dummy_image(student_dir, "01.jpg")
        self._create_dummy_image(student_dir, "02.jpg")

        # Mock detector to return exactly 1 detection per image
        dummy_crop = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
        det = DetectionResult(box=(20, 20, 140, 140), confidence=0.98, face_crop=dummy_crop)
        self.builder.detector.detect_faces = MagicMock(return_value=[det])

        rep = self.builder.process_student_folder("21CS101", "Aarav Sharma")
        self.assertEqual(rep["images_found"], 2)
        self.assertEqual(rep["valid_images"], 2)
        self.assertEqual(rep["rejected_images"], 0)
        self.assertEqual(rep["embeddings_generated"], 2)
        self.assertEqual(rep["status"], "SUCCESS")

        # Verify embedding exists in the store
        self.assertTrue(self.builder.store.exists("21CS101"))
        stored = self.builder.store.get_embedding("21CS101")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["student_name"], "Aarav Sharma")
        self.assertEqual(stored["embedding"].shape, (512,))
        # Unit norm verified
        self.assertAlmostEqual(np.linalg.norm(stored["embedding"]), 1.0, places=4)

    def test_embedding_persistence_and_reload(self):
        """Verify saving and re-reading student embeddings from disk."""
        student_dir = self.faces_dir / "21CS101"
        self._create_dummy_image(student_dir, "01.jpg")

        dummy_crop = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
        det = DetectionResult(box=(20, 20, 140, 140), confidence=0.98, face_crop=dummy_crop)
        self.builder.detector.detect_faces = MagicMock(return_value=[det])

        self.builder.process_student_folder("21CS101", "Aarav Sharma")
        self.builder.store.save()

        # Reload with fresh EmbeddingStore instance
        reloaded_store = EmbeddingStore(storage_path=self.store_path)
        self.assertEqual(len(reloaded_store), 1)
        self.assertTrue(reloaded_store.exists("21CS101"))
        rec = reloaded_store.get_embedding("21CS101")
        self.assertEqual(rec["student_name"], "Aarav Sharma")
        self.assertEqual(rec["embedding"].shape, (512,))


if __name__ == "__main__":
    unittest.main()
