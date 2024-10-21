import tensorflow as tf

# Generate some example data
X = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = tf.constant([[2.0], [4.0], [6.0], [8.0]])

# Define the model (y = W * X + b)
class LinearModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(0.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = LinearModel()

# Define the loss function (Mean Squared Error)
def loss_fn(model, X, y):
    y_pred = model(X)
    return tf.reduce_mean(tf.square(y_pred - y))

# Define the optimizer (Gradient Descent)
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, X, y)
    
    # Compute gradients
    gradients = tape.gradient(loss, [model.W, model.b])
    
    # Update parameters
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss: {loss.numpy()} W: {model.W.numpy()} b: {model.b.numpy()}")

# Output final values
print(f"Final: W = {model.W.numpy()}, b = {model.b.numpy()}")
