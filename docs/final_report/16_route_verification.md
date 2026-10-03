# Chapter 16: Route Verification & Capacity Control

## 16.1 Route Verification Objectives
Institutional bus passes are granted for specific transit corridors (e.g. Route `R-101`: "Downtown to Campus"; Route `R-102`: "Railway Station to Tech Park"). Allowing students to arbitrarily board non-assigned buses causes route overcrowding and capacity imbalances.

## 16.2 Parameterized Cross-Matching Mechanism
When a transit terminal starts, it is initialized with two persistent identifiers:
- `bus_id`: Physical vehicle registration code (e.g. `BUS-12`).
- `route`: The operational route code currently being driven (e.g. `R-101`).

During Rule 8 of `PassVerifier`, the system compares the student's assigned route against the bus's active route:
```sql
SELECT bp.route_id, br.route_code, br.route_name
FROM bus_passes bp
JOIN bus_routes br ON bp.route_id = br.id
WHERE bp.student_id = ? AND bp.status = 'ACTIVE'
```

1. **Normalized String Matching:** Route codes are normalized (stripped of whitespace, converted to uppercase) to prevent rejection caused by minor casing differences (e.g., `r-101` vs `R-101`).
2. **Deterministic Rejection:** If `student.assigned_route != terminal.active_route`, entry is immediately blocked with reason code `ROUTE_MISMATCH` and human message:  
   `"Pass is for Route R-102, but this bus is running Route R-101"`.
3. **Capacity Telemetry:** Every `ROUTE_MISMATCH` rejection is logged to SQLite, giving college transport planners data on where students are attempting to board unauthorized routes.
