import numpy as np
import tensorflow as tf

# Simulate data for the Kalman filter
def simulate_data(true_R, true_Q, num_steps):
    # True system state and measurements
    true_states = []
    measurements = []
    
    x = np.array([[0], [0], [1], [1]])  # initial true state: [x, y, vx, vy]
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    for _ in range(num_steps):
        # Process noise
        w = np.random.multivariate_normal([0, 0, 0, 0], true_Q).reshape(4, 1)
        x = F @ x + w
        
        # Measurement noise
        v = np.random.multivariate_normal([0, 0], true_R).reshape(2, 1)
        z = np.array([[x[0, 0]], [x[1, 0]]]) + v
        
        true_states.append(x)
        measurements.append(z)
        
    return np.array(true_states), np.array(measurements)

# Kalman filter implementation
def kalman_filter(measurements, F, H, R, Q, initial_state):
    num_steps = measurements.shape[0]
    n = initial_state.shape[0]
    
    # Initialize state and covariance
    x = np.copy(initial_state)
    P = np.eye(n)
    
    estimated_states = []
    
    for k in range(num_steps):
        # Predict step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update step
        z_k = measurements[k].reshape(-1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x = x_pred + K @ (z_k - H @ x_pred)
        P = (np.eye(n) - K @ H) @ P_pred
        
        estimated_states.append(x)
        
    return np.array(estimated_states)

# Loss function: Mean squared error between true states and estimated states
def loss_fn(R, Q, true_states, measurements, F, H, initial_state):
    R_matrix = tf.linalg.diag(R)
    Q_matrix = tf.linalg.diag(Q)
    
    estimated_states = kalman_filter(measurements, F, H, R_matrix, Q_matrix, initial_state)
    return tf.reduce_mean(tf.square(estimated_states[:, :2] - true_states[:, :2]))

# TensorFlow optimization
def optimize_R_Q(true_states, measurements, initial_R, initial_Q, F, H, initial_state, learning_rate=0.01, epochs=100):
    # Convert initial values to TensorFlow variables
    R = tf.Variable(initial_R, dtype=tf.float32)
    Q = tf.Variable(initial_Q, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn(R, Q, true_states, measurements, F, H, initial_state)

        gradients = tape.gradient(loss, [R, Q])
        optimizer.apply_gradients(zip(gradients, [R, Q]))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy()}, R = {R.numpy()}, Q = {Q.numpy()}")
    
    return R.numpy(), Q.numpy()

# Define constants and simulate data
np.random.seed(42)
true_R = np.array([[0.1, 0], [0, 0.1]])
true_Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])
num_steps = 100

# Simulate true states and noisy measurements
true_states, measurements = simulate_data(true_R, true_Q, num_steps)

# Kalman filter parameters
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
initial_state = np.array([[0], [0], [0], [0]])

# Initialize R and Q with initial guesses
initial_R = np.array([0.5, 0.5])
initial_Q = np.array([0.5, 0.5, 0.5, 0.5])

# Optimize R and Q using gradient descent
optimized_R, optimized_Q = optimize_R_Q(true_states, measurements, initial_R, initial_Q, F, H, initial_state)

print("Optimized R:", optimized_R)
print("Optimized Q:", optimized_Q)
