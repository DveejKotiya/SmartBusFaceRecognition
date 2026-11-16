# SmartBusFaceRecognition — Architecture

This document describes the architecture represented by the current repository. The system is an educational prototype; see the README's limitations and privacy sections before treating it as a deployment design.

## 1. Live bus-entry verification flow

```mermaid
flowchart TD
    A[Student at bus entrance] --> B[OpenCV webcam frames]
    B --> C[MTCNN face detection and landmarks]
    C --> D{Face suitable for processing?}
    D -- No --> X[Show unavailable / retry]
    D -- Yes --> E[Active liveness challenge]
    E --> F{Challenge passed?}
    F -- No --> G[Reject attempt and show reason]
    F -- Yes --> H[FaceNet InceptionResnetV1]
    H --> I[512-dimensional face embedding]
    I --> J[Embedding store / cosine similarity]
    J --> K{Known student match?}
    K -- No --> L[UNKNOWN_PERSON]
    K -- Yes --> M[PassVerifier]
    M --> N[Student active check]
    N --> O[Pass status and date validation]
    O --> P[Bus / route authorization]
    P --> Q[300-second duplicate cooldown]
    Q --> R{All checks passed?}
    R -- Yes --> S[Allow boarding]
    R -- No --> T[Deny boarding with reason]
    S --> U[SQLite entry log]
    G --> U
    L --> U
    T --> U
```

## 2. Application components

```mermaid
flowchart LR
    subgraph Runtime["Local Python application"]
      CAM[src/bus_entry_camera.py]
      SERVICE[src/core/bus_entry_service.py]
      VISION[src/vision]
      VERIFY[src/core/pass_verifier.py]
      DBL[src/database]
      UI[app.py + src/ui]
      EVAL[src/evaluation]
    end
    CAM --> SERVICE
    SERVICE --> VISION
    SERVICE --> VERIFY
    VERIFY --> DBL
    UI --> DBL
    EVAL -. evaluates .-> VISION
    EVAL -. benchmarks .-> SERVICE
    DBL --> SQLITE[(SQLite database)]
    VISION --> EMB[(Local face embeddings)]
```

## 3. Main verification responsibilities

| Component | Responsibility |
|---|---|
| `src/vision/face_detector.py` | Detect faces and extract facial landmarks with MTCNN |
| `src/vision/liveness.py` | Run the active head-movement challenge |
| `src/vision/face_recognizer.py` | Generate FaceNet facial feature vectors |
| `src/vision/embedding_store.py` | Store and compare enrolled embeddings |
| `src/core/pass_verifier.py` | Apply student, pass, date, route, and cooldown rules |
| `src/database/` | SQLite schema, connections, CRUD, and dashboard queries |
| `src/ui/` | Streamlit administration pages |

## 4. Key configured rules

- Recognition threshold: cosine similarity **0.60** by default.
- Duplicate boarding cooldown: **300 seconds**.
- Liveness is enabled and required for entry by default.
- Default terminal route values: **BUS-12 / R-101**.

Values are configurable in `src/config.py`. Check the current source before changing thresholds or using a different bus configuration.

## 5. Data handling boundary

The intended runtime is local: webcam frames are processed by the Python application, while student/pass records and entry events are handled by SQLite. Face embeddings are stored locally. Protect the database, enrollment images, and embedding files as sensitive biometric and student information. Provide a manual ID-check fallback whenever automated verification is inconclusive.
