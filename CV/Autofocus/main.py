import cv2 as cv
import numpy as np

# Create a video capture object
cap = cv.VideoCapture(0)

# Define a callback function for the trackbar
def on_trackbar(val):
    global slider_value
    slider_value = val
    print(f"Slider Value: {slider_value}")

# Create a named window
cv.namedWindow("Video Feed")

# Add a trackbar to the window, with range 0 to 100
slider_value = 0
cv.createTrackbar("Brightness", "Video Feed", 0, 100, on_trackbar)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Use the slider value to adjust something on the frame
    # (e.g., brightness, by adjusting pixel values)
    adjusted_frame = cv.convertScaleAbs(frame, alpha=1, beta=slider_value - 50)

    # Display the resulting frame
    cv.imshow("Video Feed", adjusted_frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv.destroyAllWindows()
