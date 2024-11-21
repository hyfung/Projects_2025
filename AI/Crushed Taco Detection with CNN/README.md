# Detecting Crushed Taco

## Problem Statement

- Machine packs taco into boxes
- We dont want to ship damaged tacos
- We want to classify tacos to good or bad

## Solution

- Typical binary classification problem
- Pattern detection with CNN
- Classification with binary cross-entropy
- Use a FHD camera as input
- Downsize image to reduce complexity
- Convert to grayscale to reduce complexity
- Edge detection
- Thresholding
- Now we obtained training data and real data

## Model Design

- Use any typical CNN
- Conv2D -> BatchNorm -> MaxPool -> Flatten -> Dense -> Dropout -> Dense -> Sigmoid
