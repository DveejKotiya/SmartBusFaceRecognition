# Chapter 1: Abstract

**Project Title:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Author / Candidate:** College Engineering Final Project  
**Academic Year:** 2026  

---

## 1.1 Abstract

Public and institutional campus transportation systems frequently encounter operational challenges regarding fare evasion, unauthorized boarding, transit pass counterfeiting, and boarding congestion during peak hours. Traditional bus pass verification depends heavily on manual inspection of physical laminated cards or paper receipts by bus drivers or conductors. This manual mechanism is labor-intensive, error-prone, vulnerable to card sharing among non-eligible peers, and incapable of generating automated boarding audit telemetry.

This project designs, implements, and evaluates an automated, on-device **AI-Based Face Recognition and Smart Bus Pass Verification System**. The system integrates deep convolutional neural networks for multi-task face detection (MTCNN), 512-dimensional feature embedding extraction (InceptionResnetV1 trained on VGGFace2), active challenge-response liveness detection (anti-spoofing via facial landmark yaw tracking), and a deterministic SQLite-backed pass verification rules engine. 

Operating entirely on local x86 hardware without cloud dependencies, the system enforces sequential boarding rules including facial similarity matching ($\theta = 0.60$), account status checks, date validity windows, route authorization, and a 5-minute anti-duplicate boarding cooldown window. An administrative web portal developed with Streamlit provides authorized personnel with operational oversight, pass auditing, report exports, and system diagnostics.

Empirical evaluation on the test platform demonstrates an end-to-end verification decision latency of **62.49 ms** per passenger (achieving an operational frame rate of **~28.8 FPS**), **100% detection rate** against static 2D photo presentation attacks at an evaluation overhead of just **0.0388 ms**, and **100% pass rate** across all 89 automated unit and integration tests. The platform adheres strictly to Privacy by Design principles by operating offline, keeping video frames in volatile RAM, and completely masking raw biometric vectors.
