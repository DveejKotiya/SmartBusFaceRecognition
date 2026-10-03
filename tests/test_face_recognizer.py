"""
Unit Tests for FaceRecognizer and Embedding Extraction
Smart Bus Face Recognition and Pass Verification System
"""

import unittest
import numpy as np
import torch
from PIL import Image
from src.vision.face_recognizer import FaceRecognizer, compare_embeddings


class TestFaceRecognizer(unittest.TestCase):
    """Test suite covering FaceRecognizer and compare_embeddings."""

    @classmethod
    def setUpClass(cls):
        # Use CPU for deterministic, lightweight test execution
        cls.recognizer = FaceRecognizer(device="cpu")

    def test_model_loads(self):
        """Verify InceptionResnetV1 model is loaded in evaluation mode."""
        self.assertIsNotNone(self.recognizer.model)
        self.assertFalse(self.recognizer.model.training)
        self.assertEqual(str(self.recognizer.device), "cpu")

    def test_generate_embedding_from_numpy(self):
        """Verify embedding generation from a NumPy array."""
        dummy_face = np.full((160, 160, 3), 128, dtype=np.uint8)
        embedding = self.recognizer.generate_embedding(dummy_face)

        # Check type and shape
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (512,))
        self.assertEqual(embedding.dtype, np.float32)

        # Check finite values
        self.assertTrue(np.all(np.isfinite(embedding)))

        # Check L2 unit normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_generate_embedding_from_pil(self):
        """Verify embedding generation from a PIL Image."""
        pil_face = Image.new("RGB", (160, 160), color=(100, 150, 200))
        embedding = self.recognizer.generate_embedding(pil_face)

        self.assertEqual(embedding.shape, (512,))
        self.assertTrue(np.all(np.isfinite(embedding)))

    def test_generate_embeddings_batch(self):
        """Verify batch embedding generation returns a list of 512-D embeddings."""
        face1 = np.zeros((160, 160, 3), dtype=np.uint8)
        face2 = np.ones((160, 160, 3), dtype=np.uint8) * 200
        embeddings = self.recognizer.generate_embeddings([face1, face2])

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].shape, (512,))
        self.assertEqual(embeddings[1].shape, (512,))

    def test_similarity_with_same_embedding(self):
        """Verify cosine similarity between identical embeddings is ~1.0."""
        dummy_face = np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8)
        emb1 = self.recognizer.generate_embedding(dummy_face)
        emb2 = emb1.copy()

        similarity = compare_embeddings(emb1, emb2)
        self.assertAlmostEqual(similarity, 1.0, places=4)

    def test_similarity_with_different_embeddings(self):
        """Verify similarity between opposite/orthogonal vectors behaves correctly."""
        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[1] = 1.0  # Orthogonal vector

        sim_orthogonal = compare_embeddings(v1, v2)
        self.assertAlmostEqual(sim_orthogonal, 0.0, places=4)

        v_opposite = -v1
        sim_opposite = compare_embeddings(v1, v_opposite)
        self.assertAlmostEqual(sim_opposite, -1.0, places=4)

    def test_similarity_invalid_inputs(self):
        """Verify compare_embeddings rejects invalid shapes, NaNs, and None."""
        valid_emb = np.ones(512, dtype=np.float32)

        # None inputs
        with self.assertRaises(ValueError):
            compare_embeddings(None, valid_emb)
        with self.assertRaises(ValueError):
            compare_embeddings(valid_emb, None)

        # Wrong dimensionality
        with self.assertRaises(ValueError):
            compare_embeddings(np.ones(10), valid_emb)

        # NaN / Infinite values
        nan_emb = np.ones(512, dtype=np.float32)
        nan_emb[5] = np.nan
        with self.assertRaises(ValueError):
            compare_embeddings(nan_emb, valid_emb)

        inf_emb = np.ones(512, dtype=np.float32)
        inf_emb[10] = np.inf
        with self.assertRaises(ValueError):
            compare_embeddings(valid_emb, inf_emb)


if __name__ == "__main__":
    unittest.main()
