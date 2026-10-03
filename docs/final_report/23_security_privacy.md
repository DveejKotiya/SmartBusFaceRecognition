# Chapter 23: Security, Privacy & Data Protection

## 23.1 Privacy by Design Principles
The system operates under strict **Privacy by Design** and **Data Minimization** frameworks:

1. **100% On-Premises Execution:** Video streams, face crops, landmark arrays, and neural embeddings execute completely on the local bus computer. Zero data is transmitted to cloud APIs or remote servers.
2. **Strict Biometric Vector Masking:** Raw 512-dimensional floating-point vectors are never displayed in the Streamlit UI, never written to console logs, and never exported in CSV reports. The system exposes only binary enrollment statuses (`READY`, `MISSING_IMAGES`).
3. **Ephemeral RAM Video Streams:** Live camera frames exist solely in volatile RAM as NumPy arrays. No video recordings, face crops, or temporary snapshot files are persisted to disk during transit boarding.
4. **Sanitized CSV Exports:** The CSV export query explicitly projects 9 operational columns: `entry_id, timestamp, student_id, student_name, bus_id, route, similarity, result, reason`. All biometric vectors are excluded.

## 23.2 Security Hardening Matrix
- **SQL Injection Prevention:** 100% of database interactions across `crud.py`, `dashboard_queries.py`, and `pass_verifier.py` utilize parameterized queries (`?`).
- **Path Traversal Defense:** In `build_student_embeddings.py`, folder resolution enforces `Path(student_id).name` and `is_relative_to()` validation, neutralizing directory escape payloads (`../../windows/system32`).
- **File Extension Whitelisting:** Restricted to safe image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`); executable extensions (`.exe`, `.py`, `.sh`) are rejected.
- **Git Exclusions (`.gitignore`):** Enforces strict exclusion of student photos, binary embeddings, and SQLite databases from version control.

## 23.3 Ethics & Mandatory Manual Fallback
Facial biometric systems can encounter false negatives due to extreme lighting, eyeglasses, religious headwear, or medical masks. Under no circumstances is a student stranded or denied transit arbitrarily. Conductors follow a mandatory manual fallback protocol by inspecting the student's physical College ID Card and verifying active pass validity in the admin dashboard.
