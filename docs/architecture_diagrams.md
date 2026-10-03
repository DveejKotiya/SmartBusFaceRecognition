# System Architecture & Technical Diagrams

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 10 — System Architecture Documentation  
**Date:** 2026-10-03  

---

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Hardware ["Hardware Layer (Host Laptop / PC)"]
        CAM["Webcam (720p/1080p)"]
        SCREEN["Conductor Screen / UI"]
        CPU["x86_64 Multi-Core CPU"]
    end

    subgraph Perception ["Vision & Liveness Layer"]
        MTCNN["FaceDetector (MTCNN)<br/>P-Net -> R-Net -> O-Net"]
        LIVENESS["LivenessDetector<br/>Yaw Symmetry Ratio + Variance"]
        RECOGNIZER["FaceRecognizer<br/>InceptionResnetV1 (VGGFace2)"]
        STORE["EmbeddingStore<br/>512-D Cosine Search"]
    end

    subgraph Core ["Orchestration & Business Rules"]
        SERVICE["BusEntryService<br/>Pipeline Orchestrator"]
        VERIFIER["PassVerifier<br/>6-Step Business Rules Engine"]
        CONFIG["System Config<br/>Thresholds & Cooldown"]
    end

    subgraph Storage ["Persistence Layer (Local Disk)"]
        DB[(SQLite: smart_bus.db<br/>Students, Passes, Routes, Logs)]
        PKL[("models/embeddings.pkl<br/>Binary Vectors")]
        CSV[("data/students/students.csv<br/>Student Import Source")]
    end

    subgraph Presentation ["Presentation Layer"]
        TERMINAL["OpenCV Live Terminal<br/>HUD Overlay"]
        DASHBOARD["Streamlit Admin Dashboard<br/>8 Functional Tabs"]
    end

    CAM -->|RGB Frames| SERVICE
    SERVICE --> MTCNN
    MTCNN -->|Crop + Landmarks| LIVENESS
    LIVENESS -->|Live State| SERVICE
    SERVICE -->|If Live| RECOGNIZER
    RECOGNIZER -->|512-D Float Vector| STORE
    STORE -.->|Loads on Startup| PKL
    STORE -->|Best Match & Similarity| VERIFIER
    VERIFIER <-->|Parameterized Queries| DB
    CSV -->|Student Import| DB
    VERIFIER -->|Verification Result| SERVICE
    SERVICE --> TERMINAL
    SERVICE -->|Audit Logging| DB
    TERMINAL --> SCREEN
    DB <-->|Analytical Queries| DASHBOARD
    DASHBOARD --> SCREEN
```

---

## 2. Data Flow Diagram (DFD Level 1)

```mermaid
flowchart LR
    Passenger(("Passenger"))
    Camera["Webcam Sensor"]
    Vision["Vision Pipeline<br/>(MTCNN + Liveness)"]
    Biometrics["Biometric Matcher<br/>(InceptionResnetV1)"]
    Rules["PassVerifier Engine"]
    DB[("SQLite Database")]
    Store[("Embedding Store")]
    Conductor["Conductor Terminal Display"]

    Passenger -->|Enters Bus| Camera
    Camera -->|Video Frame| Vision
    Vision -->|Action Prompt| Conductor
    Vision -->|Crop + Live State| Biometrics
    Biometrics <-->|Compare 512-D Vector| Store
    Biometrics -->|Match ID + Similarity| Rules
    Rules <-->|Query Pass & Cooldown| DB
    Rules -->|Audit Log Record| DB
    Rules -->|Decision Banner| Conductor
```

---

## 3. Face Recognition & Embedding Flow

```mermaid
sequenceDiagram
    autonumber
    actor P as Passenger
    participant C as Webcam
    participant D as FaceDetector (MTCNN)
    participant L as LivenessDetector
    participant R as FaceRecognizer (FaceNet)
    participant S as EmbeddingStore

    P->>C: Faces Camera
    C->>D: Captures BGR Frame
    D->>D: Detects Box & 5 Landmarks
    D->>L: Pass Box & Landmarks
    
    alt Spoof Detected or Challenge Timeout
        L-->>P: Crimson HUD Banner (SPOOF_SUSPECTED)
        Note over R: Face Recognition Bypassed (Saves ~46ms)
    else Live Subject Verified
        L->>R: Cropped 160x160 Face Image
        R->>R: InceptionResnetV1 Inference
        R->>R: L2 Normalization (512-D)
        R->>S: Query Candidate Vector
        S->>S: Vectorized Cosine Dot Product
        S-->>R: Best Match: Student ID & Similarity
    end
```

---

## 4. Bus Entry Decision Flowchart

```mermaid
flowchart TD
    Start(["Face Detected in Video Frame"]) --> Liveness{"Liveness Passed?"}
    
    Liveness -- No: Pending --> Prompt["Show Yellow Action Prompt<br/>(Turn Head Left/Right)"]
    Prompt --> Wait["Wait for Next Frame"]
    
    Liveness -- No: Timeout / Static --> DenySpoof["DENIED: LIVENESS_FAILED<br/>(Crimson Banner)"]
    DenySpoof --> LogDenial["Log Rejection to SQLite"]
    
    Liveness -- Yes: Verified --> Recog{"Cosine Similarity >= 0.60?"}
    
    Recog -- No --> DenyUnknown["DENIED: UNKNOWN_PERSON<br/>(Amber Banner)"]
    DenyUnknown --> LogDenial
    
    Recog -- Yes --> StudentFound{"Student in Database<br/>& Active?"}
    StudentFound -- No --> DenyStudent["DENIED: STUDENT_INACTIVE<br/>(Crimson Banner)"]
    DenyStudent --> LogDenial
    
    StudentFound -- Yes --> PassActive{"Pass Status == ACTIVE<br/>& Dates Valid?"}
    PassActive -- No --> DenyPass["DENIED: PASS_EXPIRED<br/>(Crimson Banner)"]
    DenyPass --> LogDenial
    
    PassActive -- Yes --> RouteMatch{"Pass Route == Bus Route?"}
    RouteMatch -- No --> DenyRoute["DENIED: ROUTE_MISMATCH<br/>(Crimson Banner)"]
    DenyRoute --> LogDenial
    
    RouteMatch -- Yes --> Cooldown{"Approved Entry on Route<br/>in Last 300 Seconds?"}
    Cooldown -- Yes --> DenyCooldown["DENIED: DUPLICATE_COOLDOWN<br/>(Crimson Banner)"]
    DenyCooldown --> LogDenial
    
    Cooldown -- No --> Allow["ENTRY ALLOWED: PASS_VALID<br/>(Green Banner)"]
    Allow --> LogApproval["Log Approved Entry to SQLite"]
    LogApproval --> End(["End Boarding Cycle"])
    LogDenial --> End
```

---

## 5. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    students ||--o{ bus_passes : "holds"
    students ||--o{ entry_logs : "records"
    bus_routes ||--o{ bus_passes : "valid_on"
    bus_routes ||--o{ entry_logs : "traveled_on"

    students {
        INTEGER id PK
        TEXT student_id UK "Roll Number / Identity Code"
        TEXT name "Student Full Name"
        TEXT department "Academic Department"
        INTEGER semester "Current Academic Semester"
        INTEGER is_active "1 = Active, 0 = Inactive"
        TEXT created_at "ISO-8601 Timestamp"
    }

    bus_routes {
        INTEGER id PK
        TEXT route_code UK "Unique Route Code (e.g. R-101)"
        TEXT route_name "Description of Route"
        INTEGER is_active "1 = Active, 0 = Inactive"
    }

    bus_passes {
        INTEGER id PK
        INTEGER student_id FK "References students.id"
        INTEGER route_id FK "References bus_routes.id"
        TEXT bus_id "Assigned Bus Number (e.g. BUS-12)"
        TEXT pass_start "Start Date (YYYY-MM-DD)"
        TEXT pass_end "End Date (YYYY-MM-DD)"
        TEXT status "ACTIVE or REVOKED"
        TEXT created_at "ISO-8601 Timestamp"
    }

    entry_logs {
        INTEGER id PK
        TEXT timestamp "UTC Timestamp of Boarding"
        INTEGER student_id FK "References students.id (or NULL)"
        TEXT bus_id "Bus Identifier"
        INTEGER route_id FK "References bus_routes.id"
        REAL similarity "Cosine Score (e.g. 0.88)"
        TEXT status "APPROVED or REJECTED"
        TEXT reason "Standardized ReasonCode"
    }
```

---

## 6. Physical Deployment Diagram (Laptop / Webcam Prototype)

```mermaid
graph TD
    subgraph PhysicalEnvironment ["Vehicle Entrance Physical Setup (Prototype)"]
        PASSENGER["Passenger at Bus Entrance"]
        CAM["HD USB / Integrated Camera<br/>Mounted at Eye Level (1.5m)"]
        LAPTOP["On-Bus Compute Node<br/>Windows 11 Laptop / PC (CPU Mode)"]
        DISPLAY["Operator Screen<br/>(Conductor / Driver View)"]
    end

    subgraph InternalSoftware ["Software Services Running on Node"]
        TERMINAL["OpenCV Bus Entry Camera<br/>(python src/bus_entry_camera.py)"]
        STREAMLIT["Streamlit Admin Dashboard<br/>(http://localhost:8501)"]
        DATABASE[("SQLite File<br/>data/smart_bus.db")]
        EMBEDDINGS[("Binary Vectors<br/>models/embeddings.pkl")]
    end

    PASSENGER -.->|Light Waves| CAM
    CAM -->|USB Video Interface| TERMINAL
    TERMINAL -->|Render HUD| DISPLAY
    TERMINAL <-->|Disk I/O| DATABASE
    TERMINAL <-->|Disk I/O| EMBEDDINGS
    STREAMLIT <-->|Disk I/O| DATABASE
    STREAMLIT -->|Web View| DISPLAY
```
