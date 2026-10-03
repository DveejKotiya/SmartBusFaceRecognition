# Chapter 12: Face Detection & Landmark Extraction

## 12.1 MTCNN Architecture
The system utilizes the Multi-Task Cascaded Convolutional Network (MTCNN) implemented in `facenet-pytorch`. MTCNN operates via a three-stage cascaded deep architecture:

```
[ Input Frame (480x640) ]
           │
           ▼
     [ P-Net (Proposal Network) ]
     - Fully convolutional network
     - Scans image pyramid for candidate bounding boxes
           │
           ▼
     [ R-Net (Refinement Network) ]
     - Rejects false candidates
     - Performs bounding box regression & calibration
           │
           ▼
     [ O-Net (Output Network) ]
     - Final bounding box coordinates
     - Computes 5 facial landmark locations
```

## 12.2 Five-Point Facial Landmarks
During the O-Net forward pass, MTCNN localizes 5 facial keypoints:
- $P_1 = (x_{\text{left\_eye}}, y_{\text{left\_eye}})$
- $P_2 = (x_{\text{right\_eye}}, y_{\text{right\_eye}})$
- $P_3 = (x_{\text{nose}}, y_{\text{nose}})$
- $P_4 = (x_{\text{mouth\_left}}, y_{\text{mouth\_left}})$
- $P_5 = (x_{\text{mouth\_right}}, y_{\text{mouth\_right}})$

## 12.3 Crop Preprocessing & Affine Alignment
1. **Safety Margin Padding:** A 20-pixel configurable margin is added around the raw detection box to retain natural chin and forehead contours.
2. **Coordinate Clamping:** Boundary checks clamp coordinates to the frame bounds $[0, W]$ and $[0, H]$ to eliminate indexing errors.
3. **Rescaling to 160x160:** Crops are resized to exactly $160 \times 160 \times 3$ pixels using bilinear interpolation (`cv2.INTER_AREA`).
4. **Standardization:** Pixels are normalized via standard FaceNet fixed standardization:
   $$x_{\text{norm}} = \frac{x - 127.5}{128.0}$$
