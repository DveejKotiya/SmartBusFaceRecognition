# College Presentation Slide Deck Outline (15 Slides)

**Project Title:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 10 — Final Presentation Preparation  
**Target Duration:** 15–20 Minutes (including 5-minute live demonstration)  

---

### Slide 1: Title & Team Information
- **Title:** AI-Based Face Recognition and Smart Bus Pass Verification System
- **Subtitle:** An Autonomous On-Device Biometric Access Platform for College Transit
- **Candidate Name(s) & Department:** Computer Science / Electronics Engineering
- **Project Supervisor / Guide:** [Faculty Name & Title]
- **Academic Institution:** [College Name]
- **Date:** October 2026

---

### Slide 2: Problem Statement & Motivation
- **The Transit Challenge:** High passenger volume during peak 7:30–8:30 AM boarding hours.
- **Flaws in Manual Checking:**
  - Card-sharing fraud ("pass-back" loans to unregistered students).
  - Slow visual inspection (3–6 seconds per student), causing transit delays.
  - Expired and wrong-route passes slipping through due to inspector fatigue.
  - Zero digital telemetry or attendance logs for campus transport planners.

---

### Slide 3: Analysis of Existing Systems
- **Paper & Plastic Badges:** Easily counterfeited, zero duplicate protection.
- **RFID & Smart Cards:** Fast, but verifies the *card*, not the *person* holding it.
- **Cloud-Based Face AI (AWS / Azure):** Requires high-speed continuous 5G cellular connection (fails in tunnels and suburban dead zones); recurrent API fees; biometric data uploaded to external servers.

---

### Slide 4: Proposed Solution
- **Edge-Based Autonomous Terminal:** All deep learning, verification rules, and databases operate 100% locally on the vehicle.
- **Contactless Biometric Verification:** Face recognition replaces physical badges.
- **Active Anti-Spoofing Defense:** Active head-movement challenge stops photo attacks.
- **Automated Business Rules:** Enforces active pass status, dates, routes, and a 5-minute cooldown.
- **Instant Conductor Visuals:** Color-coded HUD overlays on the driver/conductor screen.

---

### Slide 5: Project Objectives & Scope
- **Key Objectives:**
  1. Real-time face detection & embedding extraction in $<100$ ms.
  2. Sequential deterministic pass rule verification in SQLite.
  3. Elimination of pass-back fraud via a 5-minute cooldown.
  4. Lightweight liveness detection without extra deep models.
  5. Privacy by Design: 100% offline, volatile RAM video, masked vectors.
  6. Comprehensive Streamlit administration dashboard.

---

### Slide 6: Multi-Tier System Architecture
- **Layer 1 (Perception):** MTCNN face detector & InceptionResnetV1 512-D feature extractor.
- **Layer 2 (Liveness):** Active challenge-response (head yaw symmetry $\psi$) & landmark motion variance.
- **Layer 3 (Business Engine):** `PassVerifier` evaluating SQLite database rules.
- **Layer 4 (Storage):** Local SQLite transactional database (`smart_bus.db`) & binary embedding store.
- **Layer 5 (UI):** Real-time OpenCV HUD terminal + Streamlit management portal.

---

### Slide 7: AI Methodology (MTCNN & FaceNet)
- **Face Localization (MTCNN):** 3-stage deep cascade (P-Net, R-Net, O-Net) generating 160x160 aligned crops and 5 facial landmarks.
- **Deep Feature Representation (InceptionResnetV1):** VGGFace2 pretrained model combining Inception multi-scale filters with Residual skip connections.
- **Normalized 512-D Space:** Features projected onto a unit hypersphere ($L_2$ norm = 1.0).
- **Matching Metric:** Vectorized cosine dot product ($\text{Sim}(u, v) = u \cdot v$).
- **Operational Threshold:** $\theta = 0.60$ separates genuine and impostor distributions.

---

### Slide 8: Sequential Pass Verification Engine
- **Decoupled Architecture:** Probabilistic AI output feeds into deterministic business rules:
  1. *Confidence Check:* Cosine similarity $\ge 0.60$.
  2. *Account Check:* Student registered and active.
  3. *Pass Check:* Active pass record on file.
  4. *Date Check:* Current date within inclusive $[start, end]$ window.
  5. *Route Check:* Assigned route matches current bus route.
  6. *Cooldown Check:* No approved boarding on this route in previous 300 seconds.

---

### Slide 9: Anti-Spoofing & Liveness Mechanism
- **The Threat:** Attackers holding up printed color photos or smartphone selfies.
- **Zero-Dependency Solution:** Uses MTCNN landmarks to compute horizontal yaw ratio:
  $$\psi = \frac{x_{\text{nose}} - \min(x_{\text{left}}, x_{\text{right}})}{|x_{\text{right}} - x_{\text{left}}|}$$
- **Active Challenge:** Passenger prompted to turn head slightly left ($\psi \le 0.35$) or right ($\psi \ge 0.65$).
- **Passive Motion Variance:** Rejects static presentations ($\text{Var} < 10^{-6}$).
- **Compute Savings:** Rejection happens in **0.06 ms**, skipping the 46 ms Inception forward pass!

---

### Slide 10: Streamlit Administration Dashboard
- **8 Dedicated Modules:**
  - *Dashboard:* Real-time boarding feed and key performance metrics.
  - *Students & Passes:* Filterable registries with instant search.
  - *Face Data:* Enrollment tracker and one-click photo-to-embedding builder.
  - *Live Bus Entry:* In-browser camera testing terminal.
  - *Entry Logs & Reports:* Comprehensive audit history and sanitized CSV export.
  - *System Diagnostics:* Hardware specs, camera health, and SQLite integrity monitor.

---

### Slide 11: Verification & Testing Strategy
- **Rigorous Test Suite:** 89 automated tests executed via `unittest` across 9 modules:
  - Vision, Recognizer, Embedding Store, Importer, PassVerifier, Service, Dashboard, Liveness, and End-to-End Pipeline.
- **Test Integrity:** **89 / 89 Passed (100% Pass Rate)** in 10.335 seconds.
- **Boundary Verification:** Tested exact cooldown boundaries (299s vs 300s) and end-of-day inclusive pass expirations.

---

### Slide 12: Empirical Experimental Results (Measured Data)
- **Component Latencies (Host CPU):**
  - MTCNN Face Detection: **15.12 ms**
  - InceptionResnetV1 Feature Extraction: **46.12 ms**
  - Vector Similarity Search (50 candidates): **0.75 ms**
  - PassVerifier SQLite Query: **0.44 ms**
  - Liveness Verification: **0.06 ms**
  - **End-to-End Decision Time:** **62.49 ms**
- **Throughput:** **~28.8 FPS** in interleaved operational mode.
- **Biometric Evaluation:** 10,000 comparisons yielded **100% accuracy** and a **0.6813 separation margin** between genuine ($\mu = 0.8508$) and impostor ($\mu = 0.0021$) distributions.
- **Liveness Pass Rate:** **10/10 scenarios passed (100%)** against 2D presentation attacks.

---

### Slide 13: Privacy, Ethics & Limitations
- **Privacy by Design:**
  - 100% offline, zero cloud uploads.
  - Ephemeral RAM video (zero video footage or face crops saved to disk).
  - Strict vector masking (512-D floats hidden from UI and CSV exports).
- **Mandatory Manual Fallback:** Non-punitive procedure allowing conductors to verify physical College ID cards when environmental or occlusion faults occur.
- **Technical Limitations:** 2D camera cannot prevent interactive real-time deepfakes or 3D silicone masks; requires good vehicle lighting.

---

### Slide 14: Future Scope & Roadmap
- **Hardware Depth Sensors:** Incorporate active near-infrared (NIR) or Intel RealSense depth sensors for hardware-level 3D anti-spoofing.
- **Edge AI Hardware:** Port models to NVIDIA Jetson Orin Nano / TensorRT to achieve $<5$ ms inference latency.
- **Fleet Sync:** Automated Wi-Fi depot sync with central college PostgreSQL database.
- **Automated GPS Geofencing:** Automatic route switching as buses navigate transit stops.

---

### Slide 15: Conclusion & Q&A
- **Summary:** Successfully built, validated, and benchmarked an autonomous, real-time facial recognition and smart bus-pass verification system.
- **Achievements:** Sub-70 ms decision latency, 28.8 FPS video throughput, active anti-spoofing defense, 5-minute cooldown, and 100% automated test suite pass rate.
- **Open for Questions:** Thank you for your time.
