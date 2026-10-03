# System Security Review & Hardening Audit

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 9 — Security Hardening & Anti-Spoofing Audit  
**Date:** 2026-10-03  
**Status:** Audit Completed & Remediations Verified  

---

## 1. Overview

Because the Smart Bus transit verification platform processes biometric data (facial imagery, 512-dimensional feature representations) alongside sensitive student enrollment records, this security audit verifies the system against 11 essential security principles.

---

## 2. Security Audit Checklist

| Item | Security Area | Assessment & Mitigation | Verification Status |
| :-: | :--- | :--- | :-: |
| **1** | **Face Images Exposure** | Raw student photos stored in `data/students/faces/<id>/` are not published via public web routes or exposed in standard audit views. The Streamlit Face Data page displays only enrollment counts and synchronization states (`READY`, `MISSING_IMAGES`). | **VERIFIED SECURE** |
| **2** | **Embeddings Hidden from Normal UI** | Raw 512-D float vectors stored in `models/embeddings.pkl` are strictly masked. The UI renders high-level biometric statuses and similarity scores without exposing raw vector arrays. | **VERIFIED SECURE** |
| **3** | **Embeddings Excluded from CSV Reports** | The official CSV export query (`get_filtered_entry_logs` in `src/database/dashboard_queries.py`) explicitly projects 9 operational columns: `entry_id, timestamp, student_id, student_name, bus_id, route, similarity, result, reason`. Biometric vectors are absent from the export schema (validated by automated test `test_csv_export_structure`). | **VERIFIED SECURE** |
| **4** | **Sanitized Debug Logs** | Neither `print()` statements nor internal exception handlers dump floating-point biometric vectors. Only scalar cosine similarities, status strings, and truncated error messages are logged. | **VERIFIED SECURE** |
| **5** | **Parameterized SQL Queries** | 100% of database interactions across `crud.py`, `pass_verifier.py`, `student_importer.py`, and `dashboard_queries.py` utilize parameterized queries (`?` placeholders). No user inputs or parameters are concatenated directly into SQL strings. | **VERIFIED SECURE** |
| **6** | **Student ID Validation** | Roll numbers are normalized, stripped of whitespace, uppercase-standardized, and validated against alphanumeric patterns (`^[A-Za-z0-9_-]+$`) in both `student_importer.py` and `pass_verifier.py`. | **VERIFIED SECURE** |
| **7** | **Image Path Validation** | Supported file extensions are strictly limited to `SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}`. Unsupported, executable, or script extensions (`.exe`, `.py`, `.sh`, `.php`) are rejected. | **VERIFIED SECURE** |
| **8** | **Directory Traversal Prevention** | In `build_student_embeddings.py`, folder resolution executes `clean_id = Path(student_id).name` and enforces `folder.is_relative_to(faces_dir.resolve())`. Traversal payloads (`../../windows/system32`) cannot escape `data/students/faces/`. | **VERIFIED SECURE** |
| **9** | **Error Sanitization** | UI pages and user-facing terminal banners catch internal exceptions and render high-level error notices (`"Verification failure: <clean message>"`) rather than exposing raw stack traces, local filesystem structures, or credentials. | **VERIFIED SECURE** |
| **10**| **Temporary Image Cleanup** | The live entrance terminal processes video frames strictly in volatile RAM as NumPy arrays. No disk snapshots or residual temporary image files are written during streaming, eliminating orphaned temp data risks. | **VERIFIED SECURE** |
| **11**| **Version Control Exclusions** | A comprehensive `.gitignore` excludes Python bytecode (`__pycache__`), virtual environments (`.venv/`), SQLite journals (`*.db-wal`), system caches (`.DS_Store`, `Thumbs.db`), and log files. | **VERIFIED SECURE** |

---

## 3. Defense-in-Depth Model

```
[ Camera Frame ]
       │
       ▼
[ MTCNN Face Detector ] ── (Landmarks Extracted)
       │
       ▼
[ LivenessDetector ] ── (Challenge-Response & Variance Check)
       │
       ├── SPOOF DETECTED ──> Immediate Denial (LIVENESS_FAILED) & Audit Log
       │
       ▼ (LIVE ONLY)
[ FaceRecognizer (InceptionResnetV1) ] ── (512-D Embedding Extracted in RAM)
       │
       ▼
[ EmbeddingStore ] ── (In-Memory Cosine Similarity Comparison)
       │
       ▼
[ PassVerifier ] ── (Parameterized SQLite Rules & 5-Min Cooldown Check)
       │
       ▼
[ Sanitized Display & Immutable Audit Log ] ── (Zero Vectors Rendered or Exported)
```

---

## 4. Recommendations for Subsequent Deployments

1. **Hardware Security Module / TPM**: In on-bus edge hardware deployments, store `models/embeddings.pkl` and `smart_bus.db` on an encrypted volume (BitLocker / LUKS) with TPM-backed keys.
2. **Conductor Role Separation**: Implement distinct roles for Conductors (read-only verification terminal) vs. Campus Transit Administrators (student registration, pass issuance, data sync).
