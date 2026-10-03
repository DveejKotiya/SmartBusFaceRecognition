# Stage 3 Report: Face Detection, Embedding Generation & Storage
## AI-Based Face Recognition and Smart Bus Pass Verification System

---

## 1. Executive Summary

Stage 3 has been successfully implemented and verified. In this stage, we built the complete core vision engine:
1. **Face Detection** using MTCNN (`facenet-pytorch`) with bounding box clamping and 160x160 face extraction.
2. **Face Feature Extraction** using pretrained InceptionResnetV1 (`VGGFace2` weights) producing normalized 512-dimensional embedding vectors.
3. **Similarity Metric** computing bounded cosine similarity (between -1.0 and 1.0).
4. **Embedding Storage** using a file-backed persistence store (`models/embeddings.pkl`).
5. **Unit Tests & Live Verification**: 20 automated unit tests (100% pass rate), an end-to-end verification script, and a live webcam demo script.

---

## 2. Files Created & Modified

| File | Purpose | Status |
| :--- | :--- | :---: |
| `src/vision/face_detector.py` | MTCNN face detection, bounding box validation, 160x160 cropping | **Created** |
| `src/vision/face_recognizer.py` | InceptionResnetV1 feature extractor, 512-D embedding generator, cosine similarity | **Created** |
| `src/vision/embedding_store.py` | File-backed embedding persistence (`models/embeddings.pkl`), CRUD management | **Created** |
| `src/vision/__init__.py` | Vision package exports | **Modified** |
| `src/vision_demo.py` | Live webcam demonstration with bounding boxes, confidence, and embedding dimension display | **Created** |
| `tests/test_face_detector.py` | Unit tests for detector (valid/invalid images, no-face condition, bounding box format) | **Created** |
| `tests/test_face_recognizer.py` | Unit tests for recognizer (embedding shape, finiteness, unit norm, cosine similarity, invalid inputs) | **Created** |
| `tests/test_embedding_store.py` | Unit tests for storage (CRUD, persistence, reload fidelity, missing file handling) | **Created** |
| `verify_stage3.py` | End-to-end integration and sanity check script | **Created** |
| `TODO.md` | Roadmap tracking updated for Stage 3 completion | **Modified** |

---

## 3. Machine Learning Models & Technical Specifications

* **Face Detection Model**: MTCNN (Multi-task Cascaded Convolutional Networks)
  * Target output crop size: `160 x 160` pixels
  * Detection thresholds: `[0.6, 0.7, 0.7]`
  * Default confidence threshold: `0.85`
  * Face crop padding margin: `20` pixels
* **Face Embedding Model**: InceptionResnetV1 (Pretrained on `VGGFace2`)
  * Mode: Strictly evaluation / inference mode (`eval()`) with `torch.no_grad()`
  * Preprocessing: `fixed_image_standardization` ((pixel - 127.5) / 128.0)
  * Postprocessing: L2 Unit Normalization ($\|e\|_2 = 1.0$)
* **Hardware Device Used**: `CPU` (detected by PyTorch; automatically falls back if CUDA GPU is not present)
* **Embedding Dimension**: Exactly `(512,)` numeric floating-point values (`np.float32`)
* **Similarity Metric**: Cosine Similarity bounded in $[-1.0, 1.0]$:
  $$\text{Similarity}(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|}$$

---

## 4. Tests Executed & Results

All unit tests were executed using Python's `unittest` runner:
```bash
python -m unittest discover tests
```

### Test Results Breakdown:
- **`tests/test_face_detector.py`** (6 tests):
  - `test_detector_initialization`: PASSED
  - `test_prepare_rgb_image_with_numpy_bgr`: PASSED (accurate BGR-to-RGB conversion)
  - `test_prepare_rgb_image_with_pil`: PASSED
  - `test_invalid_image_inputs`: PASSED (graceful ValueError / TypeError on None/empty inputs)
  - `test_no_face_situation`: PASSED (empty list returned without crash)
  - `test_detection_result_structure_and_box_format`: PASSED (bounding box $x_2 > x_1$, $y_2 > y_1$)
- **`tests/test_face_recognizer.py`** (7 tests):
  - `test_model_loads`: PASSED (evaluation mode confirmed)
  - `test_generate_embedding_from_numpy`: PASSED (512-D, finite, L2-norm = 1.0)
  - `test_generate_embedding_from_pil`: PASSED
  - `test_generate_embeddings_batch`: PASSED
  - `test_similarity_with_same_embedding`: PASSED (self-similarity = 1.0000)
  - `test_similarity_with_different_embeddings`: PASSED (orthogonal = 0.0, opposite = -1.0)
  - `test_similarity_invalid_inputs`: PASSED (handles None, wrong dimensions, NaNs, and Infs)
- **`tests/test_embedding_store.py`** (7 tests):
  - `test_create_store_missing_file_graceful`: PASSED (empty store initialized without error)
  - `test_add_and_retrieve_embedding`: PASSED
  - `test_save_and_reload`: PASSED (exact numerical fidelity reloaded from disk)
  - `test_remove_embedding`: PASSED
  - `test_duplicate_student_id_behavior`: PASSED (deterministic record update)
  - `test_get_all`: PASSED
  - `test_invalid_embedding_inputs`: PASSED (rejects malformed arrays and missing IDs)

**Overall Unit Test Outcome**: **20 passed, 0 failed, 0 errors (0.77s)**.

---

## 5. Live Verification & Demo Script Results

### 1. Integration Verification (`verify_stage3.py`)
Executed command:
```bash
python verify_stage3.py
```
**Output Highlights**:
- FaceDetector MTCNN blank frame handled: 0 faces detected.
- FaceRecognizer embedding shape verified: `(512,)`.
- Finite numeric check: Verified no NaNs or Infs.
- L2 Unit Norm verified: `1.000000`.
- Cosine self-similarity verified: `1.000000`.
- Cross-embedding similarity verified: bounded in $[-1.0, 1.0]$.
- EmbeddingStore file persistence: Saved to `models/stage3_verify_embeddings.pkl`, reloaded, numerical match confirmed, cleaned up.
- Webcam Access: Opened `cv2.VideoCapture(0)` and captured frame `(480, 640, 3)`.

### 2. Live Webcam Demo (`src/vision_demo.py`)
Executed command:
```bash
python src/vision_demo.py --frames 5 --headless
```
**Output Highlights**:
- MTCNN initialized on CPU.
- InceptionResnetV1 initialized on CPU.
- Camera index 0 opened via DirectShow.
- Captured and processed 5 live frames at ~5.3 FPS.
- Gracefully released camera hardware and destroyed windows.

---

## 6. Problems Encountered & Fixed

1. **Direct Script Execution Import Path**:
   - *Problem*: Running `python src/vision_demo.py` caused `ModuleNotFoundError: No module named 'src'` because Python placed `src/` as the first entry in `sys.path`.
   - *Fix*: Added dynamic root resolution in `src/vision_demo.py`:
     ```python
     PROJECT_ROOT = Path(__file__).resolve().parent.parent
     if str(PROJECT_ROOT) not in sys.path:
         sys.path.insert(0, str(PROJECT_ROOT))
     ```
2. **Bounding Box Overflow Prevention**:
   - *Problem*: Raw neural network outputs for face bounding boxes can yield fractional coordinates or values exceeding the frame width/height, which can crash image array indexing.
   - *Fix*: Implemented boundary clamping in `FaceDetector.detect_faces()` using `min` and `max` constraints against the actual image dimensions before cropping.
3. **Floating-point Cosine Metric Boundaries**:
   - *Problem*: Round-off error during normalized dot product calculations can sometimes result in values like `1.0000001`.
   - *Fix*: Clamped similarity outputs safely to `[-1.0, 1.0]` using `max(-1.0, min(1.0, similarity))`.

---

## 7. Remaining Limitations & Next Steps

* **Current Limitations (By Design in Stage 3)**:
  * No student identification is performed yet (only face detection, embedding extraction, and store persistence).
  * No threshold for match/rejection has been locked in.
  * Bus pass checking and UI dashboards are intentionally deferred to subsequent stages.
* **Ready for Stage 4**:
  * **Pass Verification Rules Engine**: Compare live embedding against registered embeddings, verify active status, check route matching, check expiration dates, and apply the 5-minute anti-duplicate boarding cooldown.
