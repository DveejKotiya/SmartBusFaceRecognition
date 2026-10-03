# Anti-Spoofing & Liveness Empirical Evaluation Report

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 10 — Final Evaluation & System Verification  
**Evaluation Script:** [`src/evaluation/liveness_evaluation.py`](file:///d:/SmartBusFaceRecognition/src/evaluation/liveness_evaluation.py)  
**Date:** 2026-10-03  
**Status:** 10/10 Scenarios Verified (100% Controlled Pass Rate)  

---

## 1. Executive Summary

This report documents the empirical evaluation of the active challenge-response and passive landmark motion variance anti-spoofing mechanism implemented in `src/vision/liveness.py`. 

The system tests incoming video frames against Presentation Attack Instruments (PAIs) commonly encountered at transit vehicle doors, specifically static 2D paper photographs, smartphone screens, and uncoordinated video replays.

---

## 2. Experimental Setup & Tested Scenarios

The evaluation script ran 10 controlled attack and operational scenarios simulating real-world passenger interactions:

| # | Scenario | Presentation Category | Expected State | Observed State | Status |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | **Live Person (Turn Left)** | Bona Fide Presentation | `LIVE` | `LIVE` | **PASSED** |
| 2 | **Live Person (Turn Right)** | Bona Fide Presentation | `LIVE` | `LIVE` | **PASSED** |
| 3 | **Printed Color Photo** | Presentation Attack (2D Print) | `SPOOF_SUSPECTED` | `SPOOF_SUSPECTED` | **PASSED** |
| 4 | **Smartphone Screen Photo** | Presentation Attack (2D Screen) | `SPOOF_SUSPECTED` | `SPOOF_SUSPECTED` | **PASSED** |
| 5 | **Replayed Video Clip** | Presentation Attack (Video Replay) | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | **PASSED** |
| 6 | **Rigid Pose / Freezing** | Human Edge Case | `SPOOF_SUSPECTED` | `SPOOF_SUSPECTED` | **PASSED** |
| 7 | **Natural Facial Movement** | Bona Fide Presentation | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | **PASSED** |
| 8 | **Low-Light / Occlusion** | Environmental Edge Case | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | **PASSED** |
| 9 | **Malformed Bounding Box** | Fault Injection | `ERROR` | `ERROR` | **PASSED** |
| 10| **Multiple Faces in Frame** | Multi-Passenger Scene | `P1=LIVE, P2=INSUFFICIENT` | `P1=LIVE, P2=INSUFFICIENT_DATA` | **PASSED** |

---

## 3. Measured Performance & Latency

- **Total Scenarios Evaluated:** 10
- **Scenarios Successfully Handled:** 10 / 10
- **Pass Rate:** **100.0%**
- **Average Liveness Evaluation Latency:** **0.0388 ms per face** on host CPU

### Computational Benefit:
By calculating liveness via lightweight mathematical ratios ($\approx 0.039$ ms) before calling the heavy deep neural network (`InceptionResnetV1` $\approx 94.3$ ms), fraudulent attempts are rejected immediately, saving over **99.9% of compute cycles** during an attack.

---

## 4. Scope, Limitations & Unexamined Vectors

### 4.1 What Was Evaluated
1. **Printed Color Photos:** Static presentations on paper or cardboard with zero landmark variance ($\text{Var} < 10^{-6}$).
2. **Smartphone Screen Displays:** Static selfie photos held up to the webcam.
3. **Randomized Replay Disconnects:** Video clips whose pre-recorded head movements fail to match the randomized challenge prompt (left vs. right).
4. **Cooperative Live Passengers:** Natural progression from neutral gaze to directional head turn within the challenge window (1.5s – 3.0s).
5. **Multi-Subject Independence:** Spatial isolation verifying that a live passenger's clearance never transfers to an adjacent person holding a photo.

### 4.2 What Was NOT Evaluated (Academic Limitations)
1. **Interactive Real-Time Deepfakes:** Neural generative models capable of altering head yaw in real time responding to prompts.
2. **3D Hyper-Realistic Silicone Masks:** Physical 3D masks worn by an impostor that rotate naturally with the attacker's neck.
3. **Multi-Spectral Infrared Reflectance:** 2D RGB cameras cannot analyze skin tissue absorption or structured light depth.
4. **Adversarial Patch / Makeup Attacks:** Printed adversarial patterns designed to confuse MTCNN landmark detectors.

---

## 5. Observed Failure Modes & Safe Fallbacks

1. **Deliberate Freeze / Extreme Rigidity:**  
   If a legitimate student deliberately holds completely still without following the on-screen prompt, the session times out after 1.5–3.0 seconds and flags `SPOOF_SUSPECTED`.  
   *Mitigation:* The passenger is prompted to look at the screen and turn on the second attempt, or use manual fallback.

2. **Severe Facial Occlusion (Dark Glasses / Face Coverings):**  
   When eye landmarks cannot be localized, the system returns `INSUFFICIENT_DATA` with zero biometric matching.  
   *Mitigation:* Conductor inspects the physical student ID card and authorizes manual boarding.
