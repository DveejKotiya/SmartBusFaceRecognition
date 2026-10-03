# Pass Verification & Business Rules Engine Documentation
## Smart Bus Face Recognition and Pass Verification System

---

## 1. What PassVerifier Does

The **PassVerifier** module (`src/core/pass_verifier.py`) is the decision-making "brain" of the smart bus system. 

When a camera at the bus entrance recognizes a student's face, the AI only tells us: *"This person looks 92% like Aarav Sharma (Roll No: 21CS101)."*

The AI does **not** know:
- Does Aarav actually study at this college?
- Does he have a valid bus pass?
- Did his pass expire last week?
- Is he boarding the correct bus (Route 101 vs. Route 102)?
- Did he already board this same bus 30 seconds ago?

The `PassVerifier` takes the face recognition output, queries the SQLite database, checks all college bus rules sequentially, and outputs a final, structured decision: **ALLOWED** or **DENIED**.

---

## 2. Why Face Matching and Pass Verification are Kept Separate

In clean software architecture (and good college engineering projects), we follow the **Single Responsibility Principle (SRP)**:

| Component | Responsibility | What it does NOT care about |
| :--- | :--- | :--- |
| **Face Recognition** (`src/vision/`) | Biometrics & Computer Vision | Does not care about routes, money, expiry dates, or databases. |
| **Pass Verifier** (`src/core/`) | Business Rules & Database Checks | Does not care about convolutional neural networks, camera drivers, or image tensors. |

### Why this separation is critical:
1. **Model Upgrades**: If you switch from FaceNet to another model in the future, the pass verification logic requires zero code changes.
2. **Manual Fallback**: If face recognition fails (poor lighting, camera failure, student wearing a mask), the conductor can type the student's Roll Number directly. The *exact same* `PassVerifier` checks their pass without caring that a human typed the ID instead of a neural network.
3. **Easy Testing**: We can test all business rules (expired passes, route mismatches, duplicate boarding) instantly using simulated data without needing someone to stand in front of a webcam.

---

## 3. How an Active Pass is Determined

A student's bus pass is valid if and only if **all** of the following conditions are satisfied:

1. **Record Existence**: A pass record linked to the student's `student_id` exists in the `bus_passes` table.
2. **Account Active**: The student's account in the `students` table has `is_active = 1` (not suspended or graduated).
3. **Pass Status Active**: The pass status column in `bus_passes` equals `'ACTIVE'` (not `'EXPIRED'`, `'SUSPENDED'`, or `'REVOKED'`).
4. **Start Date Valid**: The current date is greater than or equal to `start_date` (`current_date >= start_date`). If the pass starts next month, boarding is rejected (`PASS_NOT_STARTED`).
5. **Inclusive Expiration Date**: The current date is less than or equal to `expiry_date` (`current_date <= expiry_date`).
   - **Date Convention**: We use the *inclusive end-of-day* convention. A pass expiring on `2026-10-03` remains valid through `23:59:59 UTC` on October 3rd. It becomes expired at `00:00:00` on October 4th.

---

## 4. How Route Matching Works

Each bus runs on a specific route (e.g., Bus `BUS-12` runs Route `R-101`). Each student pass is assigned to an authorized route.

When a student scans:
```
Student's Pass Route (e.g. 'R-101') == Current Bus Route (e.g. 'R-101')?
```
- If **YES**: The passenger is on the right bus $\rightarrow$ PROCEED.
- If **NO**: The passenger is boarding the wrong bus $\rightarrow$ REJECT with code `ROUTE_MISMATCH` (e.g., *"Pass is valid for Route R-102 (BUS-08), not Route R-101 (BUS-12)"*).

Route strings are normalized (trimmed and converted to uppercase) so minor differences like `"r-101 "` and `"R-101"` match smoothly.

---

## 5. How the 5-Minute Anti-Duplicate Boarding Cooldown Works

### The Real-World Problem:
A webcam captures 30 frames every second. If a student stands in front of the camera for 3 seconds while walking through the bus door, the face recognizer might detect them 15 to 30 times! Without anti-duplicate logic, the database would record 30 separate boarding logs for the same student on the same trip.

### The Solution:
We enforce a configurable **Cooldown Window** (`DEFAULT_COOLDOWN_SECONDS = 300`, which is 5 minutes).

```
Last Approved Boarding Timestamp: 10:00:00 AM
Cooldown Window: 300 seconds (5 minutes)

10:00:15 AM (Elapsed = 15s)  --> DENIED (DUPLICATE_COOLDOWN)
10:02:30 AM (Elapsed = 150s) --> DENIED (DUPLICATE_COOLDOWN)
10:04:59 AM (Elapsed = 299s) --> DENIED (DUPLICATE_COOLDOWN)
10:05:00 AM (Elapsed = 300s) --> ALLOWED (Eligible again for next journey)
```

---

## 6. Why the Database is Required for Cooldown Persistence

Some beginners try to implement duplicate prevention using a Python dictionary:
```python
# MISTAKE: In-memory dictionary
cooldown_cache = {"21CS101": timestamp}
```
**Why in-memory caching alone is flawed:**
If the bus computer reboots, the conductor switches pages in the app, or the script restarts, the in-memory cache is wiped out! A student could then re-scan immediately.

**Our Approach:**
`PassVerifier` queries the SQLite table `entry_logs`:
```sql
SELECT timestamp FROM entry_logs
WHERE student_id = ? AND route_id = ? AND status = 'APPROVED'
ORDER BY id DESC LIMIT 1;
```
Because this check is grounded in persistent SQLite storage, the cooldown window remains strictly enforced even across application reboots or script restarts.

---

## 7. Rejection Reason Codes Explained

| Reason Code | Meaning | What the Screen Displays |
| :--- | :--- | :--- |
| `PASS_VALID` | All checks passed | 🟢 **Welcome Aboard!** |
| `UNKNOWN_PERSON` | Face not found in registered embeddings | 🔴 **Face Not Recognized** |
| `LOW_CONFIDENCE` | Face similarity below threshold (e.g. < 0.60) | 🔴 **Confidence Too Low. Please look at the camera.** |
| `STUDENT_NOT_FOUND`| Recognized ID does not exist in student DB | 🔴 **Student Record Not Found** |
| `STUDENT_INACTIVE` | Student account suspended or graduated | 🔴 **Student Account Deactivated** |
| `NO_PASS` | Student exists but has no bus pass record | 🔴 **No Bus Pass Issued for this Student** |
| `PASS_INACTIVE` | Pass status is 'REVOKED' or 'SUSPENDED' | 🔴 **Bus Pass Suspended** |
| `PASS_NOT_STARTED`| Pass start date is in the future | 🔴 **Pass Not Valid Yet** |
| `PASS_EXPIRED` | Pass expiration date has passed | 🔴 **Bus Pass Expired** |
| `ROUTE_MISMATCH` | Student is boarding the wrong bus | 🔴 **Wrong Route (Assigned to Route X)** |
| `DUPLICATE_COOLDOWN`| Already boarded within last 5 minutes | 🔴 **Already Boarded (Duplicate Scan)** |
| `DATABASE_ERROR` | Database connection or query issue | 🔴 **System Error: Database Unavailable** |

---

## 8. How this Connects to the Live Bus Camera in Later Stages

In upcoming stages (Stage 5 and Stage 6):
1. **Live Camera Frame** $\rightarrow$ MTCNN detects face crop.
2. **FaceNet** $\rightarrow$ Generates 512-D embedding and finds best match from `EmbeddingStore`.
3. **`PassVerifier.verify()`** $\rightarrow$ Evaluates the result and produces `VerificationResult`.
4. **UI Banner** $\rightarrow$ If `allowed`, flashes green with student photo and chimes. If `denied`, flashes red and speaks/displays the exact rejection reason.
5. **`PassVerifier.log_verification()`** $\rightarrow$ Automatically logs the boarding to `entry_logs`.
