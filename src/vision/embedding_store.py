"""
Embedding Storage Manager
Smart Bus Face Recognition and Pass Verification System

This module provides a reliable, file-backed store for registered student facial embeddings.
Embeddings are persisted to disk using Python's standard serialization (pickle).

IMPORTANT:
Only numerical 512-dimensional embeddings and student metadata are stored.
Raw images are NOT stored here.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pickle
import numpy as np


class EmbeddingStore:
    """
    Manages persistence and retrieval of 512-dimensional face embeddings.

    Each record in the store contains:
    - student_id (int or str)
    - student_name (str)
    - embedding (np.ndarray of shape (512,) and dtype float32)
    """

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the EmbeddingStore.

        Args:
            storage_path: Path to the storage file (default: models/embeddings.pkl).
                          If the file does not exist, an empty store is initialized.
        """
        if storage_path is None:
            # Default to <project_root>/models/embeddings.pkl
            project_root = Path(__file__).resolve().parent.parent.parent
            self.storage_path = project_root / "models" / "embeddings.pkl"
        else:
            self.storage_path = Path(storage_path)

        # Internal dictionary mapping str(student_id) -> record dict
        self._store: Dict[str, Dict[str, Any]] = {}

        # Ensure directory exists and load existing embeddings if file exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def add_embedding(
        self,
        student_id: Union[int, str],
        student_name: str,
        embedding: Union[np.ndarray, List[float]]
    ) -> None:
        """
        Adds or updates a student's facial embedding in the store.

        Args:
            student_id: Unique student ID (e.g. database ID or Roll Number).
            student_name: Full name of the student.
            embedding: 512-dimensional numeric embedding vector.

        Raises:
            ValueError: If student_id/name is empty, or embedding is invalid/not 512-dimensional.
        """
        if student_id is None or str(student_id).strip() == "":
            raise ValueError("student_id cannot be None or empty.")
        if not student_name or str(student_name).strip() == "":
            raise ValueError("student_name cannot be None or empty.")

        # Convert embedding to numpy array
        try:
            emb_arr = np.asarray(embedding, dtype=np.float32).flatten()
        except Exception as e:
            raise ValueError(f"Could not convert embedding to numeric array: {e}")

        # Validate shape and finite numbers
        if emb_arr.shape != (512,):
            raise ValueError(f"Embedding must be 512-dimensional. Got shape {emb_arr.shape}.")
        if not np.all(np.isfinite(emb_arr)):
            raise ValueError("Embedding contains non-finite values (NaN or Inf).")

        key = str(student_id).strip()
        self._store[key] = {
            "student_id": student_id,
            "student_name": student_name.strip(),
            "embedding": emb_arr
        }

    def get_embedding(self, student_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a student record by student_id.

        Args:
            student_id: The ID of the student.

        Returns:
            Dict containing 'student_id', 'student_name', and 'embedding',
            or None if the student is not in the store.
        """
        key = str(student_id).strip()
        record = self._store.get(key)
        if record is None:
            return None
        # Return a copy to prevent accidental outside mutation
        return {
            "student_id": record["student_id"],
            "student_name": record["student_name"],
            "embedding": record["embedding"].copy()
        }

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all stored student embedding records.

        Returns:
            List[Dict]: List of records, each containing 'student_id', 'student_name', 'embedding'.
        """
        return [
            {
                "student_id": rec["student_id"],
                "student_name": rec["student_name"],
                "embedding": rec["embedding"].copy()
            }
            for rec in self._store.values()
        ]

    def remove_embedding(self, student_id: Union[int, str]) -> bool:
        """
        Removes a student's embedding from the store.

        Args:
            student_id: The ID of the student to remove.

        Returns:
            bool: True if the student was found and removed, False otherwise.
        """
        key = str(student_id).strip()
        if key in self._store:
            del self._store[key]
            return True
        return False

    def exists(self, student_id: Union[int, str]) -> bool:
        """
        Checks whether a student has an embedding in the store.

        Args:
            student_id: The student ID to check.

        Returns:
            bool: True if present, False otherwise.
        """
        key = str(student_id).strip()
        return key in self._store

    def save(self) -> None:
        """
        Persists the in-memory embeddings to disk.
        Uses deterministic sorted keys to ensure stable file writes.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Create sorted deterministic payload
        payload = {k: self._store[k] for k in sorted(self._store.keys())}

        with open(self.storage_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """
        Loads embeddings from the storage file on disk.
        If the file does not exist or is empty, initializes an empty store without crashing.
        """
        if not self.storage_path.exists():
            self._store = {}
            return

        try:
            if self.storage_path.stat().st_size == 0:
                self._store = {}
                return

            with open(self.storage_path, "rb") as f:
                loaded = pickle.load(f)

            if isinstance(loaded, dict):
                # Validate records during load
                valid_store: Dict[str, Dict[str, Any]] = {}
                for k, v in loaded.items():
                    if isinstance(v, dict) and "embedding" in v and "student_id" in v:
                        valid_store[str(k)] = v
                self._store = valid_store
            else:
                self._store = {}
        except Exception as e:
            # Handle corrupted file gracefully
            print(f"[WARNING] Could not load embedding store from {self.storage_path}: {e}")
            self._store = {}

    def __len__(self) -> int:
        """Returns the number of students currently stored."""
        return len(self._store)
