# Real-Time Bus Entry Recognition Guide
## Smart Bus Face Recognition and Pass Verification System

---

## 1. Overview of the Real-Time Pipeline

The **Real-Time Bus Entry Recognition** module (`src/bus_entry_camera.py` and `src/core/bus_entry_service.py`) operates live at the entrance of a college bus.

As passengers step onto the bus, the camera continuously scans the entrance, identifies registered students using deep learning, validates their pass against the SQLite database, checks route assignment, prevents duplicate boardings, and displays immediate visual feedback:

```
                  ┌────────────────────────────────────────┐
                  │          Live Webcam Feed              │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    MTCNN (Face Detection & Crop)       │
                  └───────────────────┬────────────────────┘
                                      │ 160x160 Cropped Face
                                      ▼
                  ┌────────────────────────────────────────┐
                  │  InceptionResnetV1 (512-D Embedding)   │
                  └───────────────────┬────────────────────┘
                                      │ Live Vector
                                      ▼
                  ┌────────────────────────────────────────┐
                  │   EmbeddingStore (Cosine Similarity)   │
                  │   Find Best Match >= Threshold (0.60)  │
                  └───────────────────┬────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [Recognized Student]                    [Unknown Person]
                   │                                     │
                   ▼                                     ▼
        ┌──────────────────────┐              ┌──────────────────────┐
        │ PassVerifier Engine  │              │ REJECT:              │
        │ - Student Active?    │              │ UNKNOWN PERSON       │
        │ - Pass Active?       │              │                      │
        │ - Date Range Valid?  │              │ (Orange / Yellow)    │
        │ - Route Matches Bus? │              └──────────────────────┘
        │ - Cooldown Elapsed?  │
        └──────────┬───────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    🟢 ALLOWED          🔴 DENIED
    (Green Box)         (Red Box + Reason)
```

---

## 2. How Each Component Operates

### 1. Face Detection (`FaceDetector` / MTCNN)
* Finds all human faces in each video frame.
* Returns integer bounding box coordinates `(x1, y1, x2, y2)`, detection probability confidence score (e.g. `0.98`), and crops the face resized to `160x160` RGB.

### 2. Feature Extraction (`FaceRecognizer` / InceptionResnetV1)
* Converts each cropped face into a 512-dimensional numerical vector using pretrained weights from the `VGGFace2` dataset.
* Normalizes the vector to unit L2-norm ($\|e\|_2 = 1.0$).

### 3. Identity Matching (`EmbeddingStore`)
* Computes the cosine similarity between the live embedding and every registered student's stored centroid profile.
* If the highest score is $\ge 0.60$, the student is identified.
* If below $0.60$ or no records exist, the person is marked as `UNKNOWN PERSON`.

### 4. Pass Verification (`PassVerifier`)
* Validates the student's pass in the SQLite database:
  1. Student exists and `is_active == 1`.
  2. Pass exists and status is `'ACTIVE'`.
  3. `pass_start <= current_date <= pass_end` (inclusive).
  4. Pass route matches current bus (e.g., `R-101`).
  5. Last approved boarding was more than 5 minutes (300 seconds) ago.

---

## 3. Visual Feedback Legend

| Color | Status | What the Screen Displays | Meaning |
| :--- | :--- | :--- | :--- |
| 🟢 **Green** | **ENTRY ALLOWED** | Student Name, Roll Number, Similarity Score | All checks passed. Boarding permitted. |
| 🔴 **Red** | **ENTRY DENIED** | Student Name, Roll Number, Exact Rejection Reason | Recognized student, but pass is expired, wrong route, or duplicate scan. |
| 🟡 **Amber** | **UNKNOWN PERSON** | `UNKNOWN PERSON` | Face not found in registered database. |

---

## 4. Anti-Duplicate Boarding & Database Persistence

* **The Problem**: A student standing in front of the camera for 3 seconds generates ~30 detections. We must not create 30 boarding records.
* **The Solution**: When a student is approved, exactly **one** record is written to `entry_logs`. Any subsequent scan within 5 minutes (300 seconds) on the same route triggers `DUPLICATE_COOLDOWN`.
* **Restart Resilience**: Because this check is grounded in the SQLite database table (`entry_logs`), rebooting the computer or restarting the script does not wipe out the 5-minute cooldown.

---

## 5. Performance & Responsiveness Optimizations

1. **Frame Skipping (`RECOGNITION_INTERVAL = 2`)**:
   Deep learning inference is run every 2 frames, while camera rendering remains smooth at 30 FPS.
2. **DirectShow Backend**:
   On Windows, OpenCV uses `cv2.CAP_DSHOW` to eliminate video stream startup delays.
3. **CPU Inference**:
   The pretrained InceptionResnetV1 and MTCNN pipelines run efficiently on ordinary college laptops without requiring expensive GPUs.

---

## 6. Failure & Safety Handling

* **Unknown Faces**: Never guessed or force-mapped to a student. If similarity $< 0.60$, marked as `UNKNOWN PERSON` and rejected.
* **Database Errors**: If the SQLite database is temporarily locked or disk is full, the exception is caught, logged, and marked as `DATABASE_ERROR`. The live video stream continues running without crashing.
* **Camera Disconnection**: Gracefully exits, releases the OpenCV capture device, and destroys windows.

---

## 7. Current Prototype Limitations

1. **Ambient Lighting**: Very dark environments or intense backlighting can impair MTCNN face detection.
2. **Extreme Angles**: Faces turned more than $45^\circ$ away from the camera will not yield sufficient facial features for recognition.
3. **Manual Fallback**: If face recognition fails due to a mask or lighting, the conductor uses Roll Number search (Stage 4 / Stage 8).
