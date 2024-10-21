import cv2
import numpy as np
import pprint
import json
import tf

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
