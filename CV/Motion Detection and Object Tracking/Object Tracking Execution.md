# Implementing Object Tracking 2: Execution

## Main Loop

```python

tracks = []

for frame in frames:
    # Read a frame

    # Detect objects in a frame

    # Predict tracks

    # Assign detections to tracks

    # Update tracks

    # Write tracking result

```

```matlab
tracks = table();

open(videoOut);

for frameIdx = 1:videoIn.NumFrames
    frame = read(videoIn, frameIdx);

    detections = detectCells(frame);

    tracks = predictTracks(tracks);

    [tracks, detections] = assignTracksDetections(detections, tracks);

    tracks = updateTracks(track, detections);

    writeTrackingResults(frame, tracks, videoOut);
end

close(videoOut);
```

### Detect

### Assign

- Assignment cost
- Cost of non-assignment
- Assignment optimization

### Update

- Initialize tracking parameters
- Update track estimates
- Update metadata
  - Increment all track age
  - Increment all visible track total visible count
  - Confirm tracks detected enough
  - Reset counter for consecutive frames undetected if detected
  - Increment counter for consective frames undetected
- Initialize new track
- Delete lost track
  - Visibility count over track age
    - Age threshold
    - Visibility threshold
    - Lost threshold

### Predict
