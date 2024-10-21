import cv2
import numpy as np
import pprint
import json
import tf

image = np.zeros((720, 1280, 3), dtype=np.uint8)
absolute_position = [] # Green
noisy_position = [] # Red
estimated_position = [] # Blue

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
        absolute_position.append([x,y])
        pprint.pprint(absolute_position)
        
    elif event == cv2.EVENT_LBUTTONDBLCLK:  # Right button click
        global noisy_position
        noisy_position = create_noisy_position()

def draw_points(mat, points, color):
    for point in points:
        draw_point(mat, point, color)

def draw_point(mat, point, color):
    cv2.circle(mat, point, 5, color, -1)

def draw_lines(mat, points, color):
    for i in range(1, len(points)):
        cv2.line(mat, points[i-1], points[i], color, 2)

def create_noisy_position(absolute_position):
    noisy_position = np.array(absolute_position, dtype=np.float64)
    mean = 0
    std = 30
    noise = np.random.normal(mean, std, noisy_position.shape)
    noisy_position += noise
    noisy_position = noisy_position.astype(int).tolist()
    return noisy_position

cv2.namedWindow('Frame')
cv2.createTrackbar('Q (Process Noise)', 'Frame', 1, 100, lambda x: None)
cv2.createTrackbar('R (Measurement Noise)', 'Frame', 1, 100, lambda x: None)
cv2.imshow("Frame", image)
cv2.setMouseCallback('Frame', click_event)

class KalmanFilter2D(tf.Module):
    def __init__(self, Q, R, dt=1):
        # Time step
        self.dt = dt

        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control matrix (no control input, assumed zero)
        self.B = np.zeros((4, 1))

        # Measurement matrix (we directly observe x and y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Measurement noise covariance matrix
        self.R = R ** 2 * np.eye(2)

        # Process noise covariance matrix
        q = Q ** 2
        self.Q = np.array([[q, 0, 0, 0],
                           [0, q, 0, 0],
                           [0, 0, q, 0],
                           [0, 0, 0, q]])

        # State covariance matrix
        self.P = np.eye(4)

    def predict(self):
        # Predict the next state
        self.x = self.F @ self.x + self.B

        # Predict the state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Compute Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update the state
        y = z - (self.H @ self.x)
        self.x = self.x + (K @ y)

        # Update the state covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - (K @ self.H)) @ self.P

    def get_state(self):
        return self.x
    
    def __call__(self):
        return None

while True:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    estimated_position = []
    # Compute Kalman
    process_noise_std = cv2.getTrackbarPos('Q (Process Noise)', 'Frame') * 0.01
    measurement_noise_std = cv2.getTrackbarPos('R (Measurement Noise)', 'Frame') * 0.001
    dt = 1.0
    kf = KalmanFilter2D(dt, process_noise_std, measurement_noise_std)
    
    for z in noisy_position:
        kf.predict()
        kf.update(z)
        # print("Estimated state:", kf.get_state().flatten())
        estimated_position.append([int(kf.get_state().flatten()[0]), int(kf.get_state().flatten()[1])])

    # Add points
    draw_points(image, absolute_position, GREEN)
    draw_points(image, noisy_position, RED)
    draw_points(image, estimated_position, BLUE)

    draw_lines(image, absolute_position, GREEN)
    draw_lines(image, noisy_position, RED)
    draw_lines(image, estimated_position, BLUE)

    # Render the frame
    cv2.imshow("Frame", image)
    
    key_pressed = cv2.waitKey(1)

    if key_pressed & 0xFF == ord('q'):
        break

    if key_pressed & 0xFF == ord('r'):
        # Generate new noise
        noisy_position = create_noisy_position(absolute_position)

    if key_pressed & 0xFF == ord('c'):
        estimated_position = []
        # Clear
        break

    if key_pressed & 0xFF == ord('l'):
        # Load stock path
        with open('measurements.json') as f:
            absolute_position = json.load(f)

    if key_pressed & 0xFF == ord('k'):

        estimated_position = []
        # Compute Kalman
        process_noise_std = cv2.getTrackbarPos('Q (Process Noise)', 'Frame') * 0.01
        measurement_noise_std = cv2.getTrackbarPos('R (Measurement Noise)', 'Frame') * 0.01
        dt = 1.0
        kf = KalmanFilter2D(dt, process_noise_std, measurement_noise_std)
        
        for z in noisy_position:
            kf.predict()
            kf.update(z)
            # print("Estimated state:", kf.get_state().flatten())
            estimated_position.append([int(kf.get_state().flatten()[0]), int(kf.get_state().flatten()[1])])
        

cv2.destroyAllWindows()
