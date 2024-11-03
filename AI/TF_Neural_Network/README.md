# Building AI Model in Tensorflow

## Types of Neuron (Layers)

| Name           | Description | Argument      | Purpose |
| -------------- | ----------- | ------------- | ------- |
| `Dense`        |             | `input_shape` |         |
| `Conv1D`       |             |               |         |
| `Conv2D`       |             |               |         |
| `Conv3D`       |             |               |         |
| `MaxPooling2D` |             |               |         |
|                |             |               |         |
|                |             |               |         |
|                |             |               |         |
|                |             |               |         |
|                |             |               |         |

### Dense (Fully Connected)

- The most basic layer in NN
- Connected to every neuron in previous layer
- Feedforward neural network
- Often used in hidden layer
- Classification and regression tasks
- `Dense(units=64, activation='relu')`

### Convolution (Conv1D, Conv2D, Conv3D)

- Extract features from image and spatial data
- Sliding filters (kernel) over input
- Preserve spatial relationship between pixels
- Edge, texture, objects
- `Conv2D(filters=32, kernel_size=(3,3), activation='relu')`

### Pooling (MaxPooling, AveragePooling)

- Reduce spatial dimension of feature maps
- Lowers number of parameters
- Reduce computation
- Prevent overfitting
- Max pooling -> Kernel max
- Average pooling -> Kernel average
- `MaxPooling2D(pool_size=(2, 2))`

### RNN

- Most basic current layer
- Feedback loop

### LSTM

- Enhanced RNN with long term depenedencies
- LSTM(units=50, activation='tanh')

### GRU

- Fewer parameter than LSTM
- Faster but less powerful

### Flatten

- Reshape multi-dimensional data to 1D array
- Usually done before dense layer in CNN
- Convert convolutional layer to dense layer

### Dropout

- Randomly drops a fraction of input units to zero
- Prevent overfitting by not relying too heavy on single neuron

### Batch Normalization

- Normalize the output of previous layer
- Stablize and accelerate training
- Stable distribution of input prevents internal covariate shift

### Embedding

- Used in NLP
- Represent words as a dense vector of fixed size

### Activation

- Applies an activation function
  - `relu`
  - `sigmoid`
  - `softmax`
- Improves readibility

## Types of Activation

| Name      | Description |
| --------- | ----------- |
| `relu`    |             |
| `softmax` |             |

## Design Consideration
