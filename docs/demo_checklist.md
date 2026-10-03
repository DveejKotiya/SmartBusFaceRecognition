# Final Project Demonstration Checklist & Script

**Project:** AI-Based Face Recognition and Smart Bus Pass Verification System  
**Stage:** Stage 10 — Demonstration Script & Operator Guide  
**Date:** 2026-10-03  
**Audience:** College Evaluation Committee, Project Guides & External Examiners  

---

## 1. Pre-Demo Preparation (5 Minutes Before Presentation)

- [ ] Connect laptop to power supply (ensures maximum CPU performance).
- [ ] Connect HD webcam and verify camera permissions in Windows Settings.
- [ ] Open two terminal tabs in `D:\SmartBusFaceRecognition`:
  - **Terminal 1:** For running the Live Camera Terminal (`src/bus_entry_camera.py`).
  - **Terminal 2:** For running the Streamlit Dashboard (`streamlit run app.py`).
- [ ] Prepare demonstration props:
  - **Enrolled Student:** Live presenter face (or sample student photos).
  - **Spoof Attack Prop:** Printed photo of student face on paper or displayed on phone screen.
  - **Wrong Route Prop:** Student enrolled on Route R-102 (Rohan Gupta, `21ME303`).
  - **Expired Pass Prop:** Student with expired pass (Priya Patel, `21EC202`).

---

## 2. Step-by-Step 15-Point Demonstration Script

### Step 1: Start the Streamlit Admin Dashboard
*Command:*
```bash
streamlit run app.py
```
*Action:* Open browser to `http://localhost:8501`. Highlight the clean sidebar navigation, Biometric Privacy Notice, and system health status.

---

### Step 2: Show Registered Student Records
*Action:* Navigate to **🎓 Students** page.  
*Demonstrate:* Search for `"Aarav Sharma"` or Roll Number `"21CS101"`. Show student details (Name, Department, Semester, Active status).

---

### Step 3: Show Active Bus Pass Record
*Action:* Navigate to **🎫 Bus Passes** page.  
*Demonstrate:* Filter by Status `"ACTIVE"`. Show student `21CS101` holds an active pass for Route `R-101` on Bus `BUS-12` valid until `2026-12-31`.

---

### Step 4: Show Registered Biometric Face Enrollment
*Action:* Navigate to **👤 Face Data** page.  
*Demonstrate:* Show the enrollment status card for `21CS101` (`READY`). Explain that raw 512-D vectors are stored securely in `models/embeddings.pkl` and masked from view for privacy.

---

### Step 5: Launch the Live Bus Camera Entrance Terminal
*Command (in Terminal 1):*
```bash
python src/bus_entry_camera.py --bus BUS-12 --route R-101
```
*Action:* Video window opens showing live webcam feed with top HUD bar displaying Bus ID (`BUS-12`), Active Route (`R-101`), and real-time FPS (~28-30 FPS).

---

### Step 6: Demonstrate Liveness Challenge & Action Prompt
*Action:* Present face to the camera looking straight ahead.  
*Demonstrate:*  
- Bounding box appears around face.
- Top banner displays: `LIVE CHECK: ACTION REQUIRED`.
- Bottom banner shows the prompt: `"Please turn your head slightly to your LEFT (Time remaining: 2.8s)"`.
- Explain: MTCNN calculates 5 landmarks in RAM and evaluates horizontal yaw ratio.

---

### Step 7: Complete Head Turn & Facial Recognition
*Action:* Turn head slightly in the requested direction.  
*Demonstrate:*  
- System verifies head rotation; status flips to `LIVENESS PASSED`.
- FaceNet extracts 512-D vector in ~46 ms and matches against `21CS101`.
- Terminal prints: `Face 1: Aarav Sharma (21CS101) | Sim: 0.88 | ALLOWED`.

---

### Step 8: Pass Verification Rules Engine
*Action:* Point out that recognition alone does not open the gate.  
*Demonstrate:* `PassVerifier` checks that `21CS101` has active pass status, dates are within range, route matches `R-101`, and cooldown is clear.

---

### Step 9: Show ENTRY ALLOWED Visual Feedback
*Demonstrate:*  
- Bounding box turns **Bright Green**.
- Top tag displays: `Aarav Sharma (21CS101)`.
- Large bottom HUD banner displays: `ENTRY ALLOWED (PASS_VALID)`.
- Audio/visual cue confirms transit access granted.

---

### Step 10: Show Immediate Entry Record in Logs
*Action:* Switch back to Streamlit dashboard $\rightarrow$ **📝 Entry Logs** page.  
*Demonstrate:* Click **Refresh**. Point out the newly recorded entry row showing:
- Exact UTC timestamp
- Roll Number: `21CS101`
- Bus: `BUS-12`, Route: `R-101`
- Status: `APPROVED`
- Reason: `PASS_VALID`
- Cosine Similarity: `0.88`

---

### Step 11: Demonstrate Expired Pass & Wrong Route Rejections
*Action (Expired Pass):* Simulate or scan student `21EC202` (Priya Patel, pass expired 2026-08-31).  
*Demonstrate:*  
- Bounding box turns **Crimson Red**.
- Bottom HUD displays: `ENTRY DENIED (PASS_EXPIRED)`. Entry rejected.

*Action (Wrong Route):* Simulate or scan student `21ME303` (Rohan Gupta, pass for Route R-102).  
*Demonstrate:*  
- Bounding box turns **Crimson Red**.
- Bottom HUD displays: `ENTRY DENIED (ROUTE_MISMATCH)`. Entry rejected.

---

### Step 12: Demonstrate 5-Minute Anti-Duplicate Boarding Cooldown
*Action:* Immediately present student `21CS101`'s face to the camera again within 1 minute of approval.  
*Demonstrate:*  
- Face is recognized as `Aarav Sharma`.
- Pass is valid, BUT cooldown engine detects recent entry.
- Bounding box turns **Crimson Red**.
- Bottom HUD displays: `ENTRY DENIED (DUPLICATE_COOLDOWN)`.
- Explain: Prevents pass sharing via cards or smartphone photos passed through bus windows.

---

### Step 13: Demonstrate Unknown Person Rejection & Photo Spoofing
*Action (Unknown Person):* Unregistered person steps in front of camera.  
*Demonstrate:*  
- Bounding box turns **Amber / Orange**.
- Top tag displays: `UNKNOWN PERSON`.
- Status: `ENTRY DENIED (UNKNOWN_PERSON)`. Zero student accounts accessed.

*Action (Photo Attack):* Hold up a printed photo or phone screen photo to camera.  
*Demonstrate:*  
- Landmark motion variance detector observes zero coordinate movement ($\text{Var} < 10^{-6}$).
- Challenge times out without rotation.
- HUD displays: `SPOOF SUSPECTED - ENTRY DENIED`.
- Explain: Recognition deep neural network was completely skipped, saving 74% CPU cycles.

---

### Step 14: Show Analytics & Visual Reports
*Action:* Navigate to **📈 Reports** page on the dashboard.  
*Demonstrate:*  
- Hourly boarding activity bar chart.
- Route distribution pie chart.
- Pass validity breakdown by department.

---

### Step 15: Demonstrate Sanitized CSV Audit Export
*Action:* On the **📈 Reports** page, click **Download Entry Logs CSV**.  
*Demonstrate:* Open the downloaded CSV file in Excel or Notepad. Show examiners that all 9 operational columns (`entry_id`, `timestamp`, `student_id`, `name`, `bus`, `route`, `similarity`, `result`, `reason`) are present, while **zero raw biometric vectors** or private face embeddings are exposed.

---

## 3. Concluding Demonstration Statement
*"This completes the deterministic demonstration of the AI-Based Face Recognition and Smart Bus Pass Verification System, proving real-time biometric identification, active anti-spoofing defense, and automated pass rule enforcement on local edge hardware."*
