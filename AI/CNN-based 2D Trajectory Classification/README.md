# Trajectory Classification

## Problem Statement

- Assume we have a stream of 1920x1080 and we have one object to track
- We want to know if this object is our object of interest

## Approaches

- We can collect patterns of tracjetory as training data
- Input domain
  - Images of 1920x1080
    - We probably want to downsize the image to reduce number of neurons
    - Pooling -> Conv2D -> Pooling -> Conv2D -> Flatten -> Dense -> Sigmoid
  - Labels: True, False
- Output domain
  - True, False
- Loss function
  - Binary crossentropy
- Optimizer
  - ADAM or SGD

### Dense Network

```
Using Tensorflow Keras, create a Dense network to perform binary classification on images with dimension 1920 x 1080 and single channel with 1 bit color depth, preserving the original aspect ratio, downsize input by 20x to reduce number of parameters
```

> Preprocessing

```python
import tensorflow as tf
import numpy as np

def resize_with_aspect_ratio(image, target_height, target_width):
    """
    Resize an image while preserving the aspect ratio by padding.
    """
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    scale = min(target_height / original_height, target_width / original_width)

    # Compute new dimensions
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    # Resize the image
    resized_image = tf.image.resize(image, (new_height, new_width))

    # Pad to the target dimensions
    delta_height = target_height - new_height
    delta_width = target_width - new_width
    padded_image = tf.image.pad_to_bounding_box(
        resized_image,
        offset_height=delta_height // 2,
        offset_width=delta_width // 2,
        target_height=target_height,
        target_width=target_width
    )
    return padded_image

# Example usage for preprocessing
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode to RGB
    image = resize_with_aspect_ratio(image, IMG_HEIGHT, IMG_WIDTH)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Test on an example image
image_path = "example_image.jpg"
processed_image = preprocess_image(image_path)
```

> Model Architecture

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Parameters
original_shape = (1080, 1920, 1)  # Original image dimensions
downscaled_shape = (54, 96, 1)    # Downscaled dimensions (20x reduction)
num_classes = 1                   # Binary classification

# Define the model
model = Sequential([
    Flatten(input_shape=downscaled_shape),  # Flatten downscaled image
    Dense(128, activation='relu'),          # First hidden layer
    Dropout(0.5),                           # Dropout for regularization
    Dense(64, activation='relu'),           # Second hidden layer
    Dropout(0.5),                           # Dropout for regularization
    Dense(num_classes, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()
```

```bash
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_2 (Flatten)          (None, 5184)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 128)               663680    
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_5 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 65        
=================================================================
Total params: 672,001
Trainable params: 672,001
Non-trainable params: 0
_________________________________________________________________
```

> Training

```python
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

### Convolution Network

```python

```

## Drawbacks

- Does not capture temporal information
- Can only perform classification after a track has been drawn
