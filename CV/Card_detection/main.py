import cv2
import numpy as np

def main():
    cv2.namedWindow('Image')

    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If the approximated contour has 4 vertices, it is likely a rectangle
            if len(approx) == 4:
                # Draw the rectangle on the original image
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

        cv2.imshow("Image", image)

        key_pressed = cv2.waitKey(33)
        if key_pressed & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
