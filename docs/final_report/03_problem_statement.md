# Chapter 3: Problem Statement

## 3.1 Overview of Current Operational Challenges
Institutional transit networks face several intertwined operational bottlenecks that undermine efficiency, revenue, and safety:

### 1. Manual Inspection Delays & Morning Bottlenecks
During morning peak boarding periods (typically 7:30 AM to 8:30 AM), tens of students converge at each designated bus stop within minutes. A conductor manually inspecting physical passes requires 3 to 6 seconds per passenger to check the student photo, stamp, route number, and expiration date. This creates severe boarding delays, extends bus dwell times at stops, causes route schedule slippage, and contributes to transit congestion.

### 2. Pass Sharing & Identity Impersonation
Physical transit cards can be loaned freely. An enrolled student holding an active pass for Route R-101 can hand their card to an un-enrolled roommate, who boards undetected. Conductors, working rapidly in dim morning lighting or crowded aisles, cannot meticulously match small thumbnail photos against real faces.

### 3. Route & Schedule Non-Compliance
College transit fleets operate distinct routes with calibrated seat capacities based on student enrollment. Students frequently attempt to board incorrect buses that are more convenient or arrive earlier, leading to severe overcrowding on popular routes while other buses run under-capacity. Manual conductors cannot readily cross-reference bus numbers against route codes on complex rosters.

### 4. Expired Passes & Revenue Leakage
Semester-based bus passes expire on predetermined calendar dates. Detecting an expired pass requires the conductor to read fine-print expiration dates on every boarding attempt. When conductors miss expired passes due to fatigue or hurry, the institution suffers revenue leakage.

### 5. Absence of Computerized Boarding Telemetry
Traditional paper passes generate zero real-time data. Institutional administrators cannot ascertain:
- Exactly how many students boarded Bus 12 today.
- Peak boarding stops along Route 101.
- Fraud attempts or duplicate entries.
- Real-time bus occupancy.

## 3.2 Problem Formulation
There is a pressing need for an automated edge system that can verify passenger identity and pass validity in **under 100 milliseconds**, prevent duplicate scans, reject fraudulent photo presentations, and record immutable audit logs—all while operating on affordable hardware without exposing sensitive student biometric data.
