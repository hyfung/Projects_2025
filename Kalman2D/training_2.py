import numpy as np
import tf

class KalmanFilter(tf.Module):
    def __init__(self,):
        self.position = []

        # Process noise
        self.Q = tf.variable(0)
        
        # Measurement noise
        self.R = tf.variable(0)

        # State covariance matrix
        self.P = [0, 0]

        # State vector [x, y]
        self.x = np.zeros((2, 1))

        pass

    def predict(self):
        pass

    def update(self, z):
        # 
        pass

