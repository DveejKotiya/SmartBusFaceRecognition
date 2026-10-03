"""
Vision Package
Smart Bus Face Recognition and Pass Verification System

Exposes face detection, face recognition / embedding generation,
similarity calculation, and embedding storage management.
"""

from .face_detector import FaceDetector, DetectionResult
from .face_recognizer import FaceRecognizer, compare_embeddings
from .embedding_store import EmbeddingStore

__all__ = [
    "FaceDetector",
    "DetectionResult",
    "FaceRecognizer",
    "compare_embeddings",
    "EmbeddingStore"
]
