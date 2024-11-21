# OpenCV Templates

## VideoCapture

```python
import cv2 as cv
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow("Image", img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

capture.cap()
cv2.destroyAllWindows()
```

## Edge Detection

```python

```

## Thresholding

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow('Original Image')
cv2.createTrackbar('Simple Threshold', 'Original Image', 127, 255, lambda x: None)
cv2.getTrackbarPos('Simple Threshold', 'Original Image')

while True:
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Simple Thresholding
    _, thresh_binary = cv2.threshold(image, cv2.getTrackbarPos('Simple Threshold', 'Original Image'), 255, cv2.THRESH_BINARY)
    _, thresh_binary_inv = cv2.threshold(image, cv2.getTrackbarPos('Simple Threshold', 'Original Image'), 255, cv2.THRESH_BINARY_INV)
    _, thresh_trunc = cv2.threshold(image, cv2.getTrackbarPos('Simple Threshold', 'Original Image'), 255, cv2.THRESH_TRUNC)
    _, thresh_tozero = cv2.threshold(image, cv2.getTrackbarPos('Simple Threshold', 'Original Image'), 255, cv2.THRESH_TOZERO)
    _, thresh_tozero_inv = cv2.threshold(image, cv2.getTrackbarPos('Simple Threshold', 'Original Image'), 255, cv2.THRESH_TOZERO_INV)
    # Adaptive Thresholding
    # adaptive_thresh_mean = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # adaptive_thresh_gaussian = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Binary Threshold', thresh_binary)
    cv2.imshow('Binary Inverted Threshold', thresh_binary_inv)
    cv2.imshow('Truncated Threshold', thresh_trunc)
    cv2.imshow('To Zero Threshold', thresh_tozero)
    cv2.imshow('To Zero Inverted Threshold', thresh_tozero_inv)
    # cv2.imshow('Adaptive Mean Threshold', adaptive_thresh_mean)
    # cv2.imshow('Adaptive Gaussian Threshold', adaptive_thresh_gaussian)
    # Display
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

## Masking

```python

```

## ROI

```python

```
