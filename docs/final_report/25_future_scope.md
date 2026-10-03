# Chapter 25: Future Scope & Roadmap

## 25.1 Hardware Enhancements
1. **Multi-Spectral Near-Infrared (NIR) & Depth Sensors:** Integrating depth cameras (such as Intel RealSense D435) or active NIR illumination would provide 3D facial topology mapping, rendering the system impervious to high-end video replays and silicone masks.
2. **Dedicated Edge AI Accelerators:** Deploying the pipeline onto specialized edge hardware (such as an NVIDIA Jetson Orin Nano, Google Coral TPU, or Hailo-8 accelerator) would reduce InceptionResnetV1 inference latency from 46 ms to under 5 ms, increasing throughput past 60 FPS.
3. **Automotive CAN-Bus & Turnstile Integration:** Interfacing with physical solenoid door turnstiles and audio speaker sirens via vehicle relay boards to automate gate opening upon verification.

## 25.2 Software & Network Capabilities
1. **End-of-Day Depot Wi-Fi Synchronization:** When buses enter the campus transit depot at night, terminals can automatically synchronize local SQLite `entry_logs` with a central college PostgreSQL database via mutual TLS (mTLS).
2. **Differential Embedding Updates:** Automatically push newly enrolled student embedding vectors to vehicles over delta-compressed synchronization bundles.
3. **Automated Dynamic Route Geofencing:** Interfacing with an onboard GPS receiver to automatically detect the current route and bus stop, eliminating the need for manual conductor route selection.
4. **Mobile Student App Integration:** A dedicated campus app allowing students to track their pass expiration, view boarding history, and receive renewal reminders.
