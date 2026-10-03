"""
Unit Tests for EmbeddingStore
Smart Bus Face Recognition and Pass Verification System
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.vision.embedding_store import EmbeddingStore


class TestEmbeddingStore(unittest.TestCase):
    """Test suite covering the file-backed EmbeddingStore."""

    def setUp(self):
        # Create a fresh temporary directory for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_file = Path(self.temp_dir.name) / "test_embeddings.pkl"
        self.store = EmbeddingStore(storage_path=self.store_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_dummy_embedding(self, seed: int = 42) -> np.ndarray:
        """Helper to create a deterministic 512-D unit vector."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_create_store_missing_file_graceful(self):
        """Verify initializing a store with a non-existent file doesn't crash."""
        non_existent_path = Path(self.temp_dir.name) / "does_not_exist.pkl"
        store = EmbeddingStore(storage_path=non_existent_path)
        self.assertEqual(len(store), 0)
        self.assertFalse(store.exists("101"))

    def test_add_and_retrieve_embedding(self):
        """Verify adding and retrieving a student embedding."""
        emb = self._make_dummy_embedding(1)
        self.store.add_embedding(student_id="21CS101", student_name="Aarav Sharma", embedding=emb)

        self.assertEqual(len(self.store), 1)
        self.assertTrue(self.store.exists("21CS101"))

        record = self.store.get_embedding("21CS101")
        self.assertIsNotNone(record)
        self.assertEqual(record["student_id"], "21CS101")
        self.assertEqual(record["student_name"], "Aarav Sharma")
        self.assertEqual(record["embedding"].shape, (512,))
        np.testing.assert_allclose(record["embedding"], emb, rtol=1e-5)

    def test_save_and_reload(self):
        """Verify embeddings persist to disk and can be reloaded by a new store instance."""
        emb1 = self._make_dummy_embedding(10)
        emb2 = self._make_dummy_embedding(20)

        self.store.add_embedding("S1", "Student One", emb1)
        self.store.add_embedding("S2", "Student Two", emb2)
        self.store.save()

        self.assertTrue(self.store_file.exists())

        # Create a new store pointing to the saved file
        reloaded_store = EmbeddingStore(storage_path=self.store_file)
        self.assertEqual(len(reloaded_store), 2)
        self.assertTrue(reloaded_store.exists("S1"))
        self.assertTrue(reloaded_store.exists("S2"))

        rec1 = reloaded_store.get_embedding("S1")
        np.testing.assert_allclose(rec1["embedding"], emb1, rtol=1e-5)

    def test_remove_embedding(self):
        """Verify removing a student embedding."""
        emb = self._make_dummy_embedding(5)
        self.store.add_embedding("S1", "Student One", emb)
        self.assertEqual(len(self.store), 1)

        # Remove existing
        removed = self.store.remove_embedding("S1")
        self.assertTrue(removed)
        self.assertEqual(len(self.store), 0)
        self.assertFalse(self.store.exists("S1"))

        # Remove non-existent returns False
        removed_again = self.store.remove_embedding("S1")
        self.assertFalse(removed_again)

    def test_duplicate_student_id_behavior(self):
        """Verify adding an existing student ID updates the record deterministically."""
        emb1 = self._make_dummy_embedding(1)
        emb2 = self._make_dummy_embedding(2)

        self.store.add_embedding("101", "Old Name", emb1)
        self.assertEqual(len(self.store), 1)

        # Update with new name and embedding
        self.store.add_embedding("101", "Updated Name", emb2)
        self.assertEqual(len(self.store), 1)

        rec = self.store.get_embedding("101")
        self.assertEqual(rec["student_name"], "Updated Name")
        np.testing.assert_allclose(rec["embedding"], emb2, rtol=1e-5)

    def test_get_all(self):
        """Verify get_all returns all stored records."""
        self.store.add_embedding("S1", "Student 1", self._make_dummy_embedding(1))
        self.store.add_embedding("S2", "Student 2", self._make_dummy_embedding(2))
        all_records = self.store.get_all()

        self.assertEqual(len(all_records), 2)
        ids = {r["student_id"] for r in all_records}
        self.assertEqual(ids, {"S1", "S2"})

    def test_invalid_embedding_inputs(self):
        """Verify validation prevents storing malformed embeddings."""
        # Wrong length
        with self.assertRaises(ValueError):
            self.store.add_embedding("101", "Test", np.zeros(10))

        # NaN values
        bad_emb = np.zeros(512)
        bad_emb[0] = np.nan
        with self.assertRaises(ValueError):
            self.store.add_embedding("101", "Test", bad_emb)

        # Empty name or ID
        with self.assertRaises(ValueError):
            self.store.add_embedding("", "Test", self._make_dummy_embedding(1))
        with self.assertRaises(ValueError):
            self.store.add_embedding("101", "", self._make_dummy_embedding(1))


if __name__ == "__main__":
    unittest.main()
