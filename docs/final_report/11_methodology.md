# Chapter 11: Methodology

## 11.1 Progressive Iterative Development
The project followed an engineering methodology divided into 10 progressive development stages:

1. **Stage 1 (Environment Setup):** Established validated dependencies, PyTorch CPU acceleration, OpenCV webcam connectivity, and diagnostic verification scripts.
2. **Stage 2 (Database Schema):** Designed relational DDL schema in SQLite with foreign keys, index optimizations, and initial seed routes.
3. **Stage 3 (Vision Pipeline):** Implemented MTCNN face detection, InceptionResnetV1 embedding extraction, and binary pickle persistence.
4. **Stage 4 (Pass Verification):** Built `PassVerifier` business rules engine covering pass status, expiry date boundary, route matching, and 5-minute cooldown.
5. **Stage 5 & 6 (Data Ingestion & Validation):** Created CSV validation and student onboarding pipelines with path-traversal hardening.
6. **Stage 7 (Real-Time Entry Service):** Built `BusEntryService` and OpenCV camera HUD with color-coded feedback and multi-face handling.
7. **Stage 8 (Streamlit Dashboard):** Built 8-module web portal with parameterized analytical queries and sanitized CSV reporting.
8. **Stage 9 (Anti-Spoofing & Hardening):** Implemented active challenge-response liveness detection using MTCNN landmark yaw tracking.
9. **Stage 10 (Final Testing & Evaluation):** Executed complete empirical evaluations, performance benchmarks, and presentation preparation.

## 11.2 Verification-Driven Engineering
At each stage, automated unit and integration tests were implemented before progressing. This test-driven approach ensured that adding Stage 9 anti-spoofing or Stage 8 dashboard features never broke existing Stage 3 vision or Stage 4 pass verification capabilities.
