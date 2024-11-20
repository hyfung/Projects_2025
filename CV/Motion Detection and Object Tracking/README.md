https://www.coursera.org/learn/object-tracking-and-motion-computer-vision

## Motion Detection

### Background Subtraction

- Subtract frame with background frame

### Feature Matching

-

### Template Matching

- Select a portion of image

Steps to stabilize a video

- Motion Estimation
  - $P_n = P_{n-1} + \Delta_n$
- Camera Motion Estimation
- Video Correction

### Optical Flow

- $I_xu + I_yv + I_t = 0$
  - Horn-Schunk
  - Lukas-Kanade
  - Farneback
- Differences of intensity of video frames
- Use gradient to estimate velocity of pixels

## Object Tracking

Needs tracking to recognize object detected in frames

Three main steps in the cycle

1. Assign detection to tracks
   - If it is detected and within error margin
   - If it is detected but out of error margin
   - If it is not detected, use prediction
2. Update tracks
   - Assign detection to tracks
3. Predict tracks

Unassigned detections and unassigned tracks

- Determine if is False or New detection
- Assign detection with cost function i.e. Euclidean distance
- Additional factors: Size, shape, color

And two concepts

- Detections
  - Bounding boxes
  - Centroids
  - Assignment status
- Tracks
  - Identifier
  - Number of frames
  - Estimator
    - Predicts and updates track estimate
    - Kalman Filter
      - How much to trust measurement
      - How much to trust estimation (model)

Even if object disappears, you expect to see it again in future in new position
