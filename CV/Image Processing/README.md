# Image Processing

Coursera: https://www.coursera.org/learn/dsp4/home/welcome

## Image Frequency Analysis

- Most information is in edges
- Edges are point of abrupt change in signal
- Edges are space-domain feature not captured by DFT's magnitude
- Phase alignment to reproduce edges

## Image Filtering

### Types

- IIR, FIR
- Casual, noncasual
- Lowpass
- Highpass

### Example

- Moving Average
- Gaussian Blur
- Sobel

## Image Compression

Approaches

- Domain transformation
- Block level compression
- Quantization
- Entropy coding

Pixel level compression

- Reduce number bits

## Discrete Cosine Transform

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, dctn
from PIL import Image

# Step 1: Load the image and convert to grayscale
image_path = 'test.jpg'  # Replace with your image path
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_array = np.array(image)

# Step 2: Compute the 2D DCT
dct_transformed = dctn(image_array, type=2, norm='ortho')

# Step 3: Plot the original and transformed images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# DCT Transformed Image (log-scale for better visibility)
plt.subplot(1, 2, 2)
plt.imshow(np.log1p(np.abs(dct_transformed)), cmap='gray')
plt.title('DCT Transformed (Log Scale)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

> Interactive Mode

```python
import cv2
import numpy as np
from scipy.fftpack import dct, dctn

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct_transformed = dctn(gray_img, type=2, norm='ortho')
    dct_transformed = np.log1p(np.abs(dct_transformed))
    dct_transformed = (dct_transformed - dct_transformed.min()) / (dct_transformed.max() - dct_transformed.min())
    dct_transformed = (dct_transformed * 256 - 1).astype('uint8')
    cv2.imshow('test', dct_transformed)
    cv2.waitKey(33)
```

## DCT In Image Processing

- Image is represented by pixel intensities
  - Transforms spatial information into frequency information
  - Capturues how image intensity varies over region
    - Low-frequency components = smoother variation in intensity
    - High-frequency components = sharp changes in intensity
- Concentrates on low-frequency components
  - Top-left of the transform matrix
  - High-frequency components, finer details, have lower magnitude
- Compression and quantization
- JPEG Compression
  Image is cut into 8x8 pixel blocks
  - DCT on 8x8 blocks yields 8x8 matrix of frequency coefficient
- Reconstruction

$$
Q =
\begin{bmatrix}
16 & 11 & 10 & 16 & 24 & 40 & 51 & 61 \\
12 & 12 & 14 & 19 & 26 & 58 & 60 & 55 \\
14 & 13 & 16 & 24 & 40 & 57 & 69 & 56 \\
14 & 17 & 22 & 29 & 51 & 87 & 80 & 62 \\
18 & 22 & 37 & 56 & 68 & 109 & 103 & 77 \\
24 & 35 & 55 & 64 & 81 & 104 & 113 & 92 \\
49 & 64 & 78 & 87 & 103 & 121 & 120 & 101 \\
72 & 92 & 95 & 98 & 112 & 100 & 103 & 99
\end{bmatrix}
$$

## Calculating DCT Step By Step
1. Create an 8x8 matrix $X$
2. Center the values by $X - 128$
3. Define DCT-II Transformation Matrix $C$
4. Compute 2D DCT $D = C . I . C^T$