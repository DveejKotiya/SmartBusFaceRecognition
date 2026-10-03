# Chapter 2: Introduction

## 2.1 Background & Context
College transit networks serve thousands of students daily, operating fleets of buses across diverse suburban and metropolitan routes. Managing access to these buses is critical to ensure passenger safety, prevent overcrowding, allocate route capacities appropriately, and ensure that only students who have paid requisite transit fees occupy bus seats.

Historically, transit access has been governed by physical artifacts: paper tickets, cardboard stamps, laminated badges, and magnetic strip cards. In recent years, RFID smart cards have seen limited adoption. However, physical credentials possess a fundamental architectural vulnerability: they verify the presence of the **credential**, not the **identity** of the person presenting it. Consequently, pass sharing—wherein an enrolled student loans their pass to an unregistered friend—remains widespread.

## 2.2 The Rise of Biometric Verification in Transit
Biometric identification offers a compelling solution by binding transit rights directly to human physiological characteristics. Among biometric modalities (fingerprint, iris, facial recognition), **facial recognition** is unique in being non-contact, highly hygienic, and intuitive for passengers stepping through a vehicle door.

Advances in deep convolutional neural networks (CNNs), particularly the FaceNet architecture utilizing triplet loss and deep residual networks (Inception-ResNet), have brought unconstrained face verification to high operational accuracy.

## 2.3 System Concept & Purpose
The purpose of this project is to build a complete, cohesive, end-to-end college prototype that demonstrates how modern computer vision, lightweight anti-spoofing heuristics, and relational databases can be unified into an efficient edge transit terminal. 

The system operates autonomously at the bus entrance:
1. It perceives incoming passengers via a standard webcam.
2. It verifies that a live human subject is present via an active challenge-response prompt.
3. It identifies the student through cosine similarity comparison of 512-D neural embeddings.
4. It checks pass validity, expiry, route assignment, and boarding cooldowns.
5. It outputs immediate visual feedback and logs the transaction for college administrators.
