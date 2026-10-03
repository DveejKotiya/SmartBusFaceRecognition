# Chapter 6: Proposed System

## 6.1 Architectural Overview
The proposed system introduces an autonomous, edge-computing verification architecture installed directly inside the bus entrance.

```
[ Incoming Passenger ] ──> [ Wide-Angle Webcam ]
                                  │
                                  ▼
                    [ Local Bus Computer (Python) ]
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
[ Vision Pipeline ]                                [ Business Pipeline ]
- MTCNN Face Detection                             - SQLite Database
- MTCNN 5-Point Landmarks                          - Pass Status Validator
- Active Challenge Liveness                        - Date Expiry Checker
- InceptionResnetV1 (512-D)                        - Route Matching Rule
- Cosine Similarity Matching                       - 5-Min Cooldown Engine
        │                                                   │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
                   [ Conductor HUD Display ]
          🟢 Green: ALLOWED | 🔴 Red: DENIED | 🟡 Yellow: PROMPT
                                  │
                                  ▼
                   [ Local SQLite Audit Logs ]
```

## 6.2 Key Differentiating Strengths
1. **100% Offline Edge Processing:** The system requires zero internet connection to verify passengers. All deep neural network inferences, database lookups, and logging execute locally on the bus terminal.
2. **Integrated Multi-Task Liveness:** Rather than loading a separate 100 MB anti-spoofing deep network, the system repurposes the 5 facial keypoints extracted by MTCNN to compute horizontal head yaw symmetry ($\psi$). This adds only **0.0388 ms** latency per face while rejecting static photographs.
3. **Sequential Deterministic Logic:** Security decisions are decoupled from AI matching. The AI merely provides an identity assertion and confidence score; the business engine (`PassVerifier`) applies rigid deterministic database rules to authorize boarding.
4. **Immediate Conductor Feedback:** The terminal HUD displays large color-coded banners with clear status strings (`ENTRY ALLOWED`, `PASS_EXPIRED`, `INVALID_ROUTE`, `COOLDOWN_ACTIVE`), enabling conductors to supervise boarding effortlessly from meters away.
