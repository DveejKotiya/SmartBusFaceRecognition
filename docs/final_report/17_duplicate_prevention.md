# Chapter 17: Duplicate Entry Prevention & Cooldown Architecture

## 17.1 The Card / Face Sharing Vulnerability
In a facial recognition or RFID transit terminal, a common fraud vector is the "pass-back attack":
1. Passenger A presents their face at the bus door.
2. Entry is authorized; Passenger A steps aboard.
3. Passenger A reaches their phone through an open bus window, displaying a photo of their face to Passenger B standing in line.
4. Passenger B scans the photo to gain unauthorized entry.

Even with active liveness detection, if the camera allows rapid repeated scans of the same person, double-boarding errors can occur.

## 17.2 The 5-Minute Database-Backed Cooldown Engine
To counter this, `PassVerifier` enforces an immutable temporal cooldown window:
$$\Delta t_{\text{elapsed}} = T_{\text{current}} - T_{\text{last\_approved}}$$

- **Default Cooldown Duration:** 300 seconds (5 minutes).
- **Rule:** If $\Delta t_{\text{elapsed}} < 300$ seconds on the same route, entry is denied with reason code `DUPLICATE_COOLDOWN`.

```sql
SELECT timestamp FROM entry_logs
WHERE student_id = ? AND route_id = ? AND status = 'APPROVED'
ORDER BY id DESC LIMIT 1;
```

## 17.3 Database Persistence vs. In-Memory Timers
Critically, cooldown tracking is **backed by persistent SQLite disk storage**, not volatile Python in-memory dictionaries:
- **Crash Resilience:** If the camera terminal or laptop restarts, the cooldown history remains intact.
- **Cross-Process Coordination:** Multiple background services (camera HUD and Streamlit dashboard) share the exact same cooldown state without inter-process communication bottlenecks.
- **Route Specificity:** If a student legitimately transfers from Bus 12 (Route 101) to Bus 14 (Route 103) during an interchange, the cooldown on Route 101 does not prevent them from boarding their connecting route.
