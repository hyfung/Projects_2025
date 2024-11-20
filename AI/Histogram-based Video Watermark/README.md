# Histogram-based Video Signature

## Rationale

- Input video is a 1920x1080 stream
- Using histogram to reduce dimensionality
- Converts $1920 * 1080 * 3 * time * FPS $ to 2D time series
- Apply convolution or simply do dense network to train a model

## Advantage
- Histogram is not affected by shrinking frame size
- Regardless of resolution i.e. 4K or 480P the histogram is the same
- Reduce memory usage and reduce computation
