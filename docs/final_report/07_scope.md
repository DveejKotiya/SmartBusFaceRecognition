# Chapter 7: Scope of the Project

## 7.1 In-Scope Capabilities
The scope of this project encompasses:
- Implementation of a functional real-time vision terminal using standard webcams and x86 hardware.
- Pretrained deep feature extraction utilizing InceptionResnetV1 trained on VGGFace2.
- Preprocessing and affine alignment of face crops using MTCNN.
- Storage and fast retrieval of 512-dimensional biometric feature embeddings in binary format.
- Relational schema modeling in SQLite covering students, bus routes, bus passes, and entry logs.
- Validation logic covering pass status (`ACTIVE` vs `REVOKED`), date boundaries (inclusive start and end dates), route cross-checking, and anti-duplicate boarding cooldowns.
- Active challenge-response liveness detection (head turn left/right) and landmark variance tracking.
- Web-based administration portal (Streamlit) for reporting, pass search, and system monitoring.
- Parameterized SQL security, input sanitization, and path traversal guards.

## 7.2 Out-of-Scope Elements
To maintain focus on core algorithmic, verification, and privacy objectives, the following items are explicitly marked out-of-scope for this prototype:
- Physical deployment onto automotive CAN bus hardware or turnstile solenoid locks.
- Real-time GPS vehicle tracking and dynamic geofenced stop synchronization.
- Commercial payment gateway integration for automated online student fee recharges.
- Multi-spectral hardware sensors (active infrared or time-of-flight depth cameras).
- High-availability distributed database replication across a fleet of 50+ moving vehicles.
