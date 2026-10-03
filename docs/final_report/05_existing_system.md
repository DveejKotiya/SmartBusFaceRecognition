# Chapter 5: Analysis of Existing Systems

## 5.1 Review of Current Transit Access Methods

| System Type | Mechanism | Key Strengths | Key Weaknesses |
| :--- | :--- | :--- | :--- |
| **Manual Paper / Card Passes** | Conductor visually checks printed pass. | Zero hardware cost; works everywhere. | Slow (3-6s/passenger); easy to forge; no duplicate protection; zero digital logs. |
| **Barcode / QR Code Scanners** | Passenger scans static code from card or phone screen. | Fast optical scan; digital record. | High fraud risk: QR codes can be screenshotted and shared across messaging apps instantly. |
| **RFID / Smart Card Tap** | High-frequency RFID card tapped on reader. | Rapid transaction (<1s); reliable hardware. | Pass sharing: Any person holding the card can board; cards are frequently lost or damaged. |
| **Cloud-Based Facial Systems** | Camera streams video to cloud API (AWS Rekognition / Azure Face). | Centralized model; no local computing requirement. | High recurring cloud costs; requires continuous 4G/5G mobile internet on buses; high privacy risk. |

## 5.2 Architectural Limitations of Cloud & RFID Approaches
1. **Connectivity Fragility:** Transit buses travel through cellular dead zones (tunnels, rural highways, suburban fringes). Cloud-dependent systems experience latency spikes or complete outages when connectivity drops.
2. **Privacy Vulnerability:** Transmitting unencrypted facial biometric video streams over public cellular networks creates serious data protection vulnerabilities and regulatory compliance issues.
3. **Hardware Overhead:** Fingerprint scanners require physical contact, creating hygiene issues in high-traffic transit environments, and sensor glass rapidly degrades under road vibrations.
