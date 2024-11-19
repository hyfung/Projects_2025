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

### Update

- Initialize tracking parameters
- Update track estimates
- Update metadata
- Initialize new track
- Delete lost track

### Predict
