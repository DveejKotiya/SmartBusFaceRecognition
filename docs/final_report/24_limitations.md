# Chapter 24: System Limitations

## 24.1 Technical Limitations (Academic Prototype Scope)
While the prototype demonstrates high efficacy and robust error handling, the following technical limitations must be acknowledged:

1. **2D RGB Presentation Attack Vulnerabilities:**  
   The active challenge-response mechanism reliably rejects static printed photos and static phone screens. However, sophisticated presentation attacks—such as high-frame-rate interactive deepfake puppets or synchronized tablet video playback matching the prompt—could potentially deceive a single 2D RGB camera sensor.
2. **Lighting Extremes & Direct Glare:**  
   In unconstrained transit environments, low morning lighting inside the vehicle cabin or harsh direct backlighting from the sun behind a boarding passenger can reduce MTCNN landmark confidence, causing the system to fall back to `INSUFFICIENT_DATA`.
3. **Severe Facial Occlusions:**  
   Wearing dark reflective sunglasses, heavy medical respirators, or scarves obscures critical eye or mouth keypoints, preventing accurate yaw ratio calculation and triggering manual verification.
4. **Single-Sensor Queuing Dynamics:**  
   While the pipeline supports tracking multiple faces simultaneously, optimal verification speed is achieved when boarding passengers step in front of the camera sequentially. Overlapping faces in crowded entryways can cause momentary bounding-box occlusions.
5. **Absence of Central Fleet Cloud Sync:**  
   In the current prototype, the SQLite database resides locally on each bus. Real-time fleet-wide pass revocation (e.g. if a student cancels their bus service mid-day) requires end-of-day Wi-Fi synchronization at the campus depot.
