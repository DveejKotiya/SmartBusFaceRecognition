# Chapter 9: System Architecture

## 9.1 Multi-Tier Modular Architecture
The system employs a 4-tier modular architecture designed for strict decoupling of AI perception, business rules, transactional storage, and user interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                       TIER 1: UI & HUD                      │
│   - Live Camera Terminal (OpenCV HUD)                       │
│   - Streamlit Admin Dashboard (8 Functional Pages)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│             TIER 2: ORCHESTRATION & BUSINESS RULES          │
│   - BusEntryService (Pipeline coordinator)                  │
│   - PassVerifier (Deterministic rule evaluation engine)     │
│   - Config (Central parameters & thresholds)                │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐┌──────────────▼───────────────┐
│     TIER 3: AI VISION       ││      TIER 4: DATA & DB       │
│ - FaceDetector (MTCNN)      ││ - SQLite (smart_bus.db)      │
│ - LivenessDetector (Yaw)    ││ - CRUD & Parameterized SQL   │
│ - FaceRecognizer (FaceNet)  ││ - EmbeddingStore (Pickle)    │
│ - EmbeddingStore (In-Memory)││ - CSV Importer & Validator   │
└─────────────────────────────┘└──────────────────────────────┘
```

## 9.2 Layer Separation & Modularity
- **AI Perception Layer:** Responsible solely for detecting bounding boxes, calculating facial landmark coordinates, evaluating liveness, and outputting 512-D float vectors. It possesses zero knowledge of student accounts or ticket validity.
- **Orchestration Layer (`BusEntryService`):** Connects the camera stream to the vision pipeline, checks liveness, calls biometric matching, queries `PassVerifier`, and triggers database logging.
- **Business Logic Layer (`PassVerifier`):** Pure deterministic rules engine. Given an identity assertion, it queries SQLite to evaluate pass status, route matching, expiry date boundaries, and 5-minute cooldown eligibility.
- **Data Persistence Layer:** Manages SQLite transactional storage (`students`, `bus_routes`, `bus_passes`, `entry_logs`) and binary embedding storage (`models/embeddings.pkl`).
