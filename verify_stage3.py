"""
Stage 3 Verification Script
Smart Bus Face Recognition and Pass Verification System

Executes end-to-end verification of:
1. MTCNN FaceDetector
2. FaceNet FaceRecognizer
3. EmbeddingStore persistence
4. Metric validation (shape, finiteness, cosine similarity)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from src.vision.face_detector import FaceDetector, DetectionResult
from src.vision.face_recognizer import FaceRecognizer, compare_embeddings
from src.vision.embedding_store import EmbeddingStore

def verify_all():
    print("=" * 60)
    print("  STAGE 3 COMPREHENSIVE VERIFICATION")
    print("=" * 60)

    # 1. Detector Check
    print("\n[1] Testing FaceDetector...")
    detector = FaceDetector(device="cpu")
    print(f"    - MTCNN Device: {detector.device}")
    blank_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    no_faces = detector.detect_faces(blank_frame)
    assert len(no_faces) == 0, "Blank frame should yield 0 faces"
    print("    - Blank frame handled correctly: 0 faces detected.")

    # 2. Recognizer Check
    print("\n[2] Testing FaceRecognizer & Pretrained InceptionResnetV1...")
    recognizer = FaceRecognizer(device="cpu")
    print(f"    - Model Device: {recognizer.device}")
    test_face = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
    emb = recognizer.generate_embedding(test_face)
    
    # Check shape
    assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"
    print(f"    - Embedding shape verified: {emb.shape}")
    
    # Check finite
    assert np.all(np.isfinite(emb)), "Embedding must contain finite numbers"
    print("    - Finite numeric values verified (no NaNs, no Infs).")
    
    # Check L2 unit length
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm must be ~1.0, got {norm}"
    print(f"    - Unit L2-norm verified: {norm:.6f}")

    # 3. Similarity Check
    print("\n[3] Testing Cosine Similarity...")
    sim_self = compare_embeddings(emb, emb)
    assert abs(sim_self - 1.0) < 1e-4, f"Self similarity should be 1.0, got {sim_self}"
    print(f"    - Self-similarity verified: {sim_self:.6f}")

    emb_other = recognizer.generate_embedding(np.random.randint(0, 50, (160, 160, 3), dtype=np.uint8))
    sim_other = compare_embeddings(emb, emb_other)
    assert -1.0 <= sim_other <= 1.0, f"Similarity {sim_other} out of bounds"
    print(f"    - Cross-embedding similarity: {sim_other:.4f} (bounded between -1 and 1)")

    # 4. Storage & Persistence Check
    print("\n[4] Testing EmbeddingStore Persistence...")
    test_store_path = PROJECT_ROOT / "models" / "stage3_verify_embeddings.pkl"
    if test_store_path.exists():
        test_store_path.unlink()

    store = EmbeddingStore(storage_path=test_store_path)
    store.add_embedding(student_id="TEST-001", student_name="Demo Student", embedding=emb)
    store.save()
    assert test_store_path.exists(), "Store file should exist on disk after save"
    print(f"    - Stored embedding to {test_store_path.name}")

    # Reload in new instance
    reloaded_store = EmbeddingStore(storage_path=test_store_path)
    assert len(reloaded_store) == 1, "Reloaded store should contain 1 student"
    assert reloaded_store.exists("TEST-001"), "Student TEST-001 should exist in reloaded store"
    reloaded_record = reloaded_store.get_embedding("TEST-001")
    assert reloaded_record["student_name"] == "Demo Student"
    np.testing.assert_allclose(reloaded_record["embedding"], emb, rtol=1e-5)
    print("    - Reloaded and verified embedding fidelity from disk.")

    # Cleanup test store file
    test_store_path.unlink()
    print("    - Cleaned up temporary test store file.")

    # 5. Webcam Check
    print("\n[5] Testing Webcam Access...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    opened = cap.isOpened()
    if opened:
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None, "Frame read failed"
        print(f"    - Webcam successfully opened and captured frame of size {frame.shape}")
    else:
        print("    - Webcam not available at index 0.")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    verify_all()
