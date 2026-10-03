# Chapter 10: Technology Stack Rationale

## 10.1 Technology Choices and Rationale

### 1. Python as Core Language
Python was selected due to its dominant computer vision ecosystem, mature deep learning bindings, rich mathematical tooling (NumPy), and rapid web UI prototyping frameworks (Streamlit). Python enables complex AI pipelines to be integrated with databases and camera feeds in clean, beginner-friendly syntax.

### 2. PyTorch & facenet-pytorch
`facenet-pytorch` provides pure PyTorch implementations of MTCNN and InceptionResnetV1 with pretrained weights on the massive VGGFace2 dataset (3.3 million face images across 9,000+ identities). It runs natively in evaluation mode (`eval()`) with TorchScript JIT optimization, eliminating the need to install TensorFlow or compile native C++ modules.

### 3. MTCNN vs. Haar Cascades vs. YOLO
- *Haar Cascades:* Fast, but prone to extreme false positives in poor transit lighting and fails completely under slight head yaw or tilt.
- *YOLOv8:* Excellent object detector, but requires custom landmark post-processing heads.
- *MTCNN (Selected):* Jointly detects face bounding boxes, bounding box regression offsets, and 5 facial keypoints in a single cascaded pipeline, directly providing the landmarks required for our active anti-spoofing logic.

### 4. SQLite as Edge Database
Transit buses operate in moving environments where maintaining client-server database servers (like PostgreSQL or MySQL) introduces excessive background memory overhead, network configuration complexity, and risk of database corruption on abrupt bus power-off. SQLite is serverless, zero-configuration, ACID-compliant, stored in a single cross-platform file, and supports up to 100,000 queries per second on local SSDs.

### 5. Streamlit for Admin Dashboard
Streamlit allows developing interactive, responsive web applications in pure Python without writing separate JavaScript, HTML, or CSS code. It provides built-in data table caching, dynamic filters, file download widgets, and metrics cards.
