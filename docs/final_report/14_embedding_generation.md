# Chapter 14: Embedding Generation & Persistence

## 14.1 Offline Enrollment Pipeline
Enrolling a student into the biometric recognition store proceeds through an offline batch pipeline implemented in `src/vision/build_student_embeddings.py`:

```
[ Student Photo Directory (data/students/faces/<id>/) ]
                     │
                     ▼
       [ MTCNN Face Localization ]
   (Extract 160x160 crop for each photo)
                     │
                     ▼
       [ InceptionResnetV1 Forward Pass ]
     (Extract 512-D vector per image)
                     │
                     ▼
     [ Intra-Subject Vector Averaging ]
               _      1  N
               e_i = ──  ∑ e_{i, j}
                      N j=1
                     │
                     ▼
          [ L2 Re-Normalization ]
                e_norm = e / ||e||_2
                     │
                     ▼
        [ Binary Serialization ]
         (models/embeddings.pkl)
```

## 14.2 Vector Averaging Benefits
Averaging vectors extracted across 3 to 5 photographs captured under differing lighting conditions and slight pose angles significantly suppresses photographic noise, shadows, and transient facial expressions, producing a centroid reference vector that is more resilient during daily transit boarding.

## 14.3 In-Memory Lookup & Pickle Persistence
`EmbeddingStore` (`src/vision/embedding_store.py`) maintains an internal dictionary mapping `student_id -> {"student_id": id, "student_name": name, "embedding": np.ndarray}`.
- **Fast Startup:** On system launch, the binary dictionary is unpickled into RAM once.
- **Vectorized NumPy Search:** Candidate probe embeddings are broadcast across all stored embeddings simultaneously using matrix multiplication (`np.dot`), achieving a search time of just **0.75 ms for 50 enrolled identities**.
- **Corruption Recovery:** If the pickle file is missing or corrupted, `EmbeddingStore` catches `UnpicklingError`, initializes a clean in-memory state, and logs an alert without crashing the host application.
