"""
Unit Tests for FaceDetector
Smart Bus Face Recognition and Pass Verification System
"""

import unittest
import numpy as np
from PIL import Image
from src.vision.face_detector import FaceDetector, DetectionResult


class TestFaceDetector(unittest.TestCase):
    """Test suite covering the FaceDetector module."""

    @classmethod
    def setUpClass(cls):
        # Force CPU device for deterministic, lightweight unit testing
        cls.detector = FaceDetector(
            image_size=160,
            margin=10,
            min_face_size=20,
            confidence_threshold=0.8,
            device="cpu"
        )

    def test_detector_initialization(self):
        """Verify FaceDetector initializes properly."""
        self.assertIsNotNone(self.detector.mtcnn)
        self.assertEqual(self.detector.image_size, 160)
        self.assertEqual(str(self.detector.device), "cpu")

    def test_prepare_rgb_image_with_numpy_bgr(self):
        """Verify OpenCV BGR image is converted to RGB array."""
        bgr_image = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)  # Pure Blue in BGR
        rgb_image = self.detector._prepare_rgb_image(bgr_image, is_bgr=True)
        self.assertEqual(rgb_image.shape, (100, 100, 3))
        # Blue in BGR becomes Red channel 0 in RGB
        self.assertEqual(rgb_image[0, 0, 0], 0)
        self.assertEqual(rgb_image[0, 0, 2], 255)

    def test_prepare_rgb_image_with_pil(self):
        """Verify PIL image input is accepted and converted to RGB."""
        pil_img = Image.new("RGB", (80, 80), color=(10, 20, 30))
        rgb_image = self.detector._prepare_rgb_image(pil_img)
        self.assertEqual(rgb_image.shape, (80, 80, 3))
        self.assertEqual(tuple(rgb_image[0, 0]), (10, 20, 30))

    def test_invalid_image_inputs(self):
        """Verify invalid inputs raise appropriate exceptions."""
        with self.assertRaises(ValueError):
            self.detector.detect_faces(None)

        with self.assertRaises(TypeError):
            self.detector.detect_faces("invalid_file_path.jpg")

        with self.assertRaises(ValueError):
            self.detector.detect_faces(np.array([]))

        with self.assertRaises(ValueError):
            self.detector.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8))

        with self.assertRaises(ValueError):
            self.detector.detect_faces(np.zeros((50,), dtype=np.uint8))

    def test_no_face_situation(self):
        """Verify a blank image returns an empty list (no crash)."""
        blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
        results = self.detector.detect_faces(blank_image)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_detection_result_structure_and_box_format(self):
        """Verify DetectionResult container properties and bounding box constraints."""
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        result = DetectionResult(
            box=(10, 20, 110, 120),
            confidence=0.98,
            face_crop=dummy_crop
        )

        # Check types and properties
        self.assertIsInstance(result.box, tuple)
        self.assertEqual(len(result.box), 4)
        x1, y1, x2, y2 = result.box
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)

        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

        self.assertEqual(result.face_crop.shape, (160, 160, 3))


if __name__ == "__main__":
    unittest.main()
