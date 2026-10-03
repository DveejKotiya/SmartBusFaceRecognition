# Chapter 8: Software & Hardware Requirements

## 8.1 Functional Requirements (FR)
- **FR-1 (Face Detection):** The system shall detect one or more human faces in incoming video frames with confidence $\ge 0.85$.
- **FR-2 (Landmark Extraction):** The system shall extract 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`) for each detected face.
- **FR-3 (Liveness Check):** The system shall issue an active directional head-turn prompt and verify head rotation before granting entry.
- **FR-4 (Embedding Extraction):** The system shall generate a normalized 512-dimensional vector for each validated live face crop.
- **FR-5 (Identity Matching):** The system shall match the candidate vector against enrolled embeddings using cosine similarity with operational threshold $\theta = 0.60$.
- **FR-6 (Pass Verification):** The system shall verify student enrollment, active pass status, inclusive validity dates, and route assignment in SQLite.
- **FR-7 (Duplicate Cooldown):** The system shall deny boarding if the same student has an approved entry on this route within the preceding 300 seconds.
- **FR-8 (Audit Logging):** The system shall log every accepted and rejected boarding attempt with timestamp, similarity, and standardized reason code.
- **FR-9 (Visual Feedback):** The terminal display shall provide instant color-coded status overlays (Green, Crimson, Amber).
- **FR-10 (Admin Dashboard):** The administrative web UI shall provide student search, pass monitoring, log inspection, and sanitized CSV exports.

## 8.2 Non-Functional Requirements (NFR)
- **NFR-1 (Performance):** Total decision latency shall not exceed 100 ms per face on CPU.
- **NFR-2 (Privacy):** Raw 512-D vectors shall never be displayed in the UI or written to CSV exports.
- **NFR-3 (Reliability):** Video streams shall not crash if the database becomes locked or unavailable.
- **NFR-4 (Security):** Database access shall use 100% parameterized SQL queries to prevent SQL injection.

## 8.3 Hardware & Software Specification

| Component | Minimum Specification | Tested Prototype Configuration |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i3 / AMD Ryzen 3 (Quad Core) | AMD64 Multi-Core (AuthenticAMD) |
| **Memory (RAM)** | 4 GB DDR4 | 16 GB DDR4 |
| **Storage** | 2 GB available SSD/HDD | Local NVMe SSD |
| **Camera** | 720p 30 FPS USB Webcam | Integrated HD Webcam (640x480 capture) |
| **Operating System**| Windows 10/11 or Ubuntu Linux 20.04+ | Windows 11 Build 10.0.26300 |
| **Python** | Python 3.9+ | Python 3.14.0 AMD64 |
| **Deep Learning** | PyTorch >= 2.0.0 | PyTorch 2.14.0+cpu |
| **Vision Libraries**| OpenCV >= 4.8.0, facenet-pytorch >= 2.5.0 | OpenCV 5.0.0, facenet-pytorch 2.5.3 |
| **Database** | SQLite 3 | SQLite 3.50.4 |
