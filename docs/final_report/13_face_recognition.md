# Chapter 13: Face Recognition & Similarity Matching

## 13.1 InceptionResnetV1 Architecture
For biometric feature extraction, the system employs **InceptionResnetV1** pretrained on the **VGGFace2** dataset. The model combines two architectural innovations:
1. **Inception Multi-Scale Convolutions:** Concurrently processes $1 \times 1$, $3 \times 3$, and $5 \times 5$ filter banks to capture multi-scale facial details (e.g. eye corners, skin textures, jawlines).
2. **Residual Skip Connections:** Adds identity shortcut mappings ($x + F(x)$) enabling very deep neural networks to be trained without vanishing gradients.

The network terminates in an average pooling layer and a dense projection layer producing a 512-dimensional continuous floating-point vector:
$$e \in \mathbb{R}^{512}, \quad \|e\|_2 = 1.0$$

## 13.2 Cosine Similarity Metric
Because embeddings are $L_2$-normalized onto the 512-dimensional unit hypersphere, the cosine similarity between candidate embedding $u$ and enrolled embedding $v$ simplifies directly to their inner dot product:

$$\text{Sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \sum_{k=1}^{512} u_k v_k$$

- **Range:** $[-1.0, 1.0]$.
- **Identical Faces:** $\text{Sim} \approx 1.0$.
- **Different Identities (Impostor):** $\text{Sim} \approx 0.0$ to $0.20$.
- **Operational Threshold:** $\theta = 0.60$.
  - If $\text{Sim}(u, v_{\text{best}}) \ge 0.60$, the system asserts a facial match with the corresponding student.
  - If $\text{Sim}(u, v_{\text{best}}) < 0.60$, the system classifies the passenger as `UNKNOWN_PERSON`.
