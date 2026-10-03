# Chapter 4: Project Objectives

## 4.1 Primary Objectives
The central goal of this project is to develop and evaluate a functional, real-time, on-device AI system that automates the verification of college bus passes using facial recognition.

Specifically, the system aims to:
1. **Automate Passenger Identification:** Detect faces from a live webcam feed and accurately identify registered students using deep feature representations (FaceNet InceptionResnetV1).
2. **Enforce Deterministic Business Validation:** Connect biometric identity to an SQLite relational database to verify pass status, validity dates, authorized routes, and boarding cooldowns.
3. **Prevent Card Sharing (Anti-Duplicate Cooldown):** Enforce a 5-minute (300-second) cooldown window preventing any student pass from being used multiple times in rapid succession.
4. **Implement Active Anti-Spoofing (Liveness Detection):** Design an efficient liveness verification mechanism that prevents presentation attacks using static photos or phone screens without adding extra heavy deep learning dependencies.
5. **Provide an Administrative Dashboard:** Build a modern, accessible web interface (Streamlit) for student directory management, pass status tracking, audit log inspection, CSV reports, and hardware monitoring.
6. **Ensure Privacy by Design:** Guarantee 100% on-device processing, volatile RAM-only video frame handling, and strict masking of raw 512-dimensional biometric vectors.
7. **Incorporate Conductor Manual Fallback:** Provide a non-discriminatory, humane procedure allowing conductors to verify physical College ID cards when environmental or technical faults occur.

## 4.2 Engineering Success Metrics
- **Verification Latency:** End-to-end decision time $< 100$ ms per face presentation.
- **Operational Frame Rate:** Minimum 20 FPS video preview on multi-core CPU.
- **Test Suite Integrity:** 100% pass rate on all automated unit and integration tests.
- **Biometric Security:** Zero false accepts at operational threshold $\theta = 0.60$ under benchmark evaluation.
