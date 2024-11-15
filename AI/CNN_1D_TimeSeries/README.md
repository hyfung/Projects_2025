# CNN or Regular Network for 1D Series

## Problem Definiton

- We want to recognize the red waveform using a classifier
- Dimensions
  - `t` = range(100)
  - `x` = range(256)
- There are 2 approaches
  - Binary classification for simplicity
    - If it is red, return 0
    - If it is not red, return 1
  - Multi-class classification for fun
    - Red = 0
    - Yellow = 1
    - Green = 2
    - Cyan = 3
    - Blue = 4

## Training Data Preparation

### Helper Function

> Slide your mouse through mat to create sample waveform

```python
import cv2
import numpy as np

mouse_is_down = False
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Left button clicked at ({x}, {y})")
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        # print(f"Left button clicked at ({x}, {y})")
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        points.append((x, y))
        # print(f"Mouse moved to ({x}, {y})")

window_name = "Drawing board"
image = np.zeros((100, 100), dtype=np.uint8)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


while True:
    image = np.zeros((100, 100), dtype=np.uint8)
    for point in points:
        image[point[1]][point[0]] = 255
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Quit
        break
    if key == ord('r'):
        # Reset
        points = []
    if key == ord('e'):
        data = np.argmax(image, axis=0)
        data = 100 - data
        # If any of the data point is missing which is 100
        # Take average of 2 neighbours to pad
        for i in range(1, 99):
            if data[i] == 100:
                data[i] = int((data[i-1] + data[i+1]) / 2)
        print(list(data))
        points = []

cv2.destroyAllWindows()
```

> Plotting different waveforms

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 10 random 1D arrays, each with 100 elements
data = []

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot each array
for i, array in enumerate(data):
    plt.plot(array, label=f'Array {i+1}')

# Add labels, legend, and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('10 One-Dimensional Arrays')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()

# Show the plot
plt.show()
```

### True Data (Red)

- Hand draw 10 base pattern
- Apply gaussian noise to data points

### False Data (Non-red)

- Hand draw 10 base pattern
- Apply gaussian noise to data points

## Network Design

### Dense

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer

model = Sequential([
    InputLayer(100),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

```

### Conv1D

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Input, InputLayer, Reshape, Dense, Flatten

model = Sequential([
    InputLayer(100),
    Reshape((100, 1)),
    Conv1D(filters=10, kernel_size=3, strides=1, activation='relu'),
    Flatten(),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()
```
