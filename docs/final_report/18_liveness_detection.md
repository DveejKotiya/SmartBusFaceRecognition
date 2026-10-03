# Chapter 18: Anti-Spoofing & Liveness Detection

## 18.1 Vulnerabilities of 2D Facial Recognition
Standard 2D facial recognition systems extract embeddings from pixel intensity arrays without distinguishing between a living human subject and a Presentation Attack Instrument (PAI). The most prevalent attacks at vehicle entrances include:
1. **Printed Color Photos:** High-resolution photographs printed on paper or cardboard.
2. **Smartphone Screen Photos:** Digital selfies displayed on high-brightness OLED or LCD displays.
3. **Prerecorded Video Replays:** Looping video clips of an enrolled student displayed on a tablet.

## 18.2 Active Challenge-Response Architecture
To defeat these attacks without adding expensive depth cameras or heavy neural networks, the system implements an **Active Challenge-Response** mechanism (`src/vision/liveness.py`) leveraging the 5 facial keypoints inherently calculated by MTCNN:
- $P_1 = (x_{\text{left\_eye}}, y_{\text{left\_eye}})$
- $P_2 = (x_{\text{right\_eye}}, y_{\text{right\_eye}})$
- $P_3 = (x_{\text{nose}}, y_{\text{nose}})$

### Normalized Horizontal Yaw Ratio:
$$\psi = \frac{x_{\text{nose}} - \min(x_{\text{left\_eye}}, x_{\text{right\_eye}})}{|x_{\text{right\_eye}} - x_{\text{left\_eye}}|}$$

```
    Turn Left (Subject's Left)       Neutral Frontal Gaze       Turn Right (Subject's Right)
            (psi <= 0.35)             (0.40 <= psi <= 0.60)            (psi >= 0.65)
              [ Nose < Eye ]               [ Nose Centered ]              [ Nose > Eye ]
```

## 18.3 Verification Sequence & State Machine
1. **Initial Assessment (`INSUFFICIENT_DATA`):** System detects a face and assigns a randomized challenge (`TURN_LEFT` or `TURN_RIGHT`) with a yellow HUD prompt:  
   `"Please turn your head slightly to your LEFT (Time remaining: 2.8s)"`.
2. **Passive Motion Variance Check:** Tracks landmark variance across 6 rolling frames. If variance is near-zero ($\text{Var} < 10^{-6}$), the subject is flagged as a static 2D photograph (`SPOOF_SUSPECTED`).
3. **Challenge Satisfaction (`LIVE`):** Passenger rotates head beyond threshold within the 3.0-second timeout window. The state flips to `LIVE` ($confidence = 0.95$), and the pipeline immediately proceeds to InceptionResnetV1 face recognition.
4. **Timeout Expiration (`SPOOF_SUSPECTED`):** If no rotation occurs within 3.0 seconds, entry is rejected with reason code `LIVENESS_FAILED` and a crimson HUD banner.

## 18.4 Spatial Multi-Person Isolation
`LivenessDetector` tracks sessions across frames using bounding-box Intersection-over-Union (IoU > 0.35). Each face in the field of view has an isolated state machine. A live passenger completing their turn will never inadvertently authorize an adjacent person holding up a photo.
