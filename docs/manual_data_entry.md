# Manual Student & Bus Pass Data Entry Workflow
## Smart Bus Face Recognition and Pass Verification System

---

## 1. Overview

In this system, all student profiles, bus-pass details, and face photos are **entered and curated manually by you**. No complex registration UI or automatic enrollment is required.

You control two things:
1. **The Spreadsheet / CSV File**: [`data/students/students.csv`](file:///d:/SmartBusFaceRecognition/data/students/students.csv) (student details, bus numbers, routes, dates).
2. **The Face Images Directory**: [`data/students/faces/<student_id>/`](file:///d:/SmartBusFaceRecognition/data/students/faces/) (photos of each student).

---

## 2. Step-by-Step Workflow

Follow these 7 steps whenever you want to enroll a new student or update an existing one:

### STEP 1: Open the Student CSV File
Open [`data/students/students.csv`](file:///d:/SmartBusFaceRecognition/data/students/students.csv) in Microsoft Excel, Google Sheets, VS Code, or Notepad.

### STEP 2: Add or Edit the Student Row
Add a row with the following 9 columns:

| Column | Description | Example |
| :--- | :--- | :--- |
| `student_id` | Unique Roll Number / Student ID | `21CS101` |
| `name` | Student's Full Name | `Aarav Sharma` |
| `department` | Department / Branch | `Computer Science` |
| `semester` | Current semester number | `6` |
| `bus_id` | Assigned Bus Number | `BUS-12` |
| `route` | Assigned Route Code | `R-101` |
| `pass_start` | Pass start date (`YYYY-MM-DD`) | `2026-07-01` |
| `pass_end` | Pass expiration date (`YYYY-MM-DD`) | `2026-12-31` |
| `pass_status`| Status: `ACTIVE`, `INACTIVE`, or `EXPIRED` | `ACTIVE` |

*Example line in `students.csv`:*
```csv
21CS101,Aarav Sharma,Computer Science,6,BUS-12,R-101,2026-07-01,2026-12-31,ACTIVE
```

### STEP 3: Create the Student's Face Folder
Create a folder inside `data/students/faces/` with the **exact same name as `student_id`**:
```
data/students/faces/21CS101/
```

### STEP 4: Add Face Photos
Put 3 to 10 clear photos of the student into their folder:
```
data/students/faces/21CS101/
├── 01.jpg
├── 02.jpg
├── 03.jpg
└── 04.jpg
```
*(See [`docs/face_image_requirements.md`](file:///d:/SmartBusFaceRecognition/docs/face_image_requirements.md) for photo tips).*

### STEP 5: Import CSV Data into SQLite Database
Run this terminal command:
```bash
python -m src.data.student_importer
```
*What this does:*
- Reads `students.csv`.
- Validates dates, IDs, and statuses.
- Inserts or updates the student, route, and bus pass in the SQLite database (`data/smart_bus.db`).

### STEP 6: Generate Face Embeddings
Run this terminal command:
```bash
python -m src.vision.build_student_embeddings
```
*What this does:*
- Scans `data/students/faces/<student_id>/`.
- Detects the face in each photo with MTCNN.
- Rejects photos with 0 faces or more than 1 face.
- Extracts 512-dimensional embeddings using FaceNet.
- Averages valid photos into an optimal biometric profile.
- Saves the embeddings to `models/embeddings.pkl`.

### STEP 7: Run the Verification Terminal or Demo
The live bus entrance will now recognize the student and verify their pass against your manual entries!

---

## 3. Directory Layout Example

```
SmartBusFaceRecognition/
├── data/
│   └── students/
│       ├── students.csv             <-- Edit student records here
│       ├── students_template.csv    <-- Template reference
│       └── faces/
│           ├── 21CS101/             <-- Folder matches student_id
│           │   ├── 01.jpg
│           │   ├── 02.jpg
│           │   └── 03.jpg
│           ├── 21EC202/
│           │   ├── 01.jpg
│           │   └── 02.jpg
│           └── 21ME303/
│               ├── 01.jpg
│               └── 02.jpg
```

---

## 4. Update Rules & Behavior

### If you change a student's information:
- Changing **Name, Department, Semester, Bus, Route, Pass Dates, or Pass Status**:
  - Edit `students.csv` and run `python -m src.data.student_importer`.
  - You do **NOT** need to regenerate face embeddings! Their face biometrics remain valid.

### If you add or replace student photos:
- Add or replace `.jpg` files in `data/students/faces/<student_id>/`.
- Run `python -m src.vision.build_student_embeddings`.

### If you change a Student's ID:
- In database systems, the `student_id` (Roll Number) is the **primary key**.
- If a student's ID changes (e.g. from `TEMP-01` to `21CS101`), rename their folder in `data/students/faces/` and update `students.csv`, then run both commands:
  ```bash
  python -m src.data.student_importer
  python -m src.vision.build_student_embeddings
  ```
