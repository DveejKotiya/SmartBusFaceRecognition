# Face Image Quality Guidelines & Requirements
## Smart Bus Face Recognition and Pass Verification System

---

## 1. Overview

FaceNet and MTCNN rely on clear facial features (eyes, nose bridge, jawline, mouth) to generate distinct 512-dimensional embeddings. The quality and variety of photos you manually place into `data/students/faces/<student_id>/` will directly influence how accurately the bus entrance camera recognizes the student.

---

## 2. Key Image Guidelines

### 1. Exactly One Person Per Photo
* **DO**: Use photos containing only the registered student.
* **DON'T**: Use group photos or selfies with friends in the background. The embedding builder will automatically reject images containing more than one face to prevent corrupting the student's biometric profile.

### 2. High Clarity & Focus
* **DO**: Use sharp, well-focused photos where the student's eyes and facial contours are clearly visible.
* **DON'T**: Use extremely blurry, heavily compressed, pixelated, or motion-blurred pictures.

### 3. Balanced Lighting
* **DO**: Capture photos with even, natural lighting across the face.
* **DON'T**: Use photos with harsh backlighting (e.g. standing directly in front of a bright window or sunlight where the face is completely in shadow).

### 4. Facial Visibility & Obstructions
* **DO**: Ensure the face is uncovered and looking generally toward the camera.
* **DON'T**: Avoid heavy sunglasses, thick scarves covering the chin/nose, or face masks during enrollment photos. Prescription glasses are fine if they do not cause intense glare over the eyes.

### 5. Variety of Angles and Expressions
* To help the system recognize the student in real-world bus boarding conditions, include photos with slight variations:
  - 1-2 straight-on neutral expression photos.
  - 1-2 slightly smiling or natural expression photos.
  - 1-2 photos with a slight head turn (e.g., $10^\circ$ to the left and right).
  - 1-2 photos under different indoor/outdoor ambient lighting.

---

## 3. Recommended Number of Images

* **Recommended Range**: **3 to 10 photos** per student.
* *Note on Accuracy*: While having 3 to 10 good photos allows the system to compute a robust centroid embedding (averaging out minor noise), no fixed number of photos can guarantee 100% recognition accuracy under adverse conditions (such as darkness or extreme angles). The system's **Manual Fallback** feature is always available if face recognition is inconclusive.

---

## 4. Privacy & Consent Notice

* **Ethical and College Policy**: Always obtain explicit permission and consent from students before collecting, storing, and using their photographs in this biometric attendance project.
* **File Formats Supported**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
