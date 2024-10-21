# Support Vector Machine

## Principle
Mathematical function to transform input data to igher-dimensional space for easier classification

Makes complex relationship become more linear

Hyperplane to separate the data

### Kernel

Function
- Transform data
- Implicit mapping

#### Linear
- K(x,x′)=x⋅x′
- When data is linearly separatable
- Straight line

#### Polynomial
- K(x,x′)=(γx⋅x′+r)d
- Similarity of vectors in polynomial space
- Decision boundary is not linear but polynomial
- γ, r, and dd are kernel parameters
- Degree of polynomial

#### Sigmoid
- K(x,x′)=tanh(γx⋅x′+r)
- Activation function in neural networks

#### Radial Basis Function
- K(x,x′)=exp(−γ∥x−x′∥2)
- No clear linear separation
- Maps data into infinite-dimensional space
- γ is a free parameter that defines the influence of individual data points.

## Parameters

### C (Regularization Parameter)
- Trade-off between low-error training data and minimize margin
- Smaller C -> More margin violation, smoother decision boundary
- Larger C -> Less margin violation, might overfit

### Gamma (Kernel coefficient, RBF, Poly, Sig)
- Influence of each data point
- Lower Gamma -> Data points far apart considered similar
- Higher Gamma -> Only data points very close will be considered similar

### Degree (Poly)
- Degree of polynomial
- Higher degree -> More complex decision boundary

### Coef0 (Poly, Sig)
- Controls influence of higher-order term
- How model handles curvature of decision boundary
- Adds flexibility how model adjust for high order component
- Used for fine-tuning

### Tolerance
- Stopping criterion
- How long to run before changes is below certain threshold

### Max Iterations
- Maximum number of iteration to run

## Training

## Hyperparameter Tuning

### Grid Search Cross Validation

## Classification
