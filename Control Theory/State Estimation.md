# State Estimation

## Central Limit Theorem

## Point Set Registration Problem

## Iterative Closest Point Algorithm

## Jacobian Matrix

- Matrix of first order partial deriatives
- Linearization of Non-linear function
- Linearization error
- Approximation of a non-linear function
- State transition model and measurement model

## Kalman Filter

- Error state Extended Kalman Filter
- Unscented Kalman Filter
- Extended Kalman Filter

| .         | EKF                      | ES-EKF                    | UKF                 |
| --------- | ------------------------ | ------------------------- | ------------------- |
| Principle | Full State Linearization | Error State Linearization | Unscented Transform |
| Accuracy  | Good                     | Better                    | Best                |
| Jacobian  | Required                 | Required                  | Not Required        |
| Speed     | Slightly Faster          | Slightly Faster           | Slightly Slower     |

## Keywords

- Least squares
- Max likelihood
- Squared error criteria
- Least error criteria
- Sum of squared error
- Measurement noise variance

## Method of Lease Square

- Ordinary
- Weighted
  - Sensor fusion
- Recursive

### Squared Error Criterion

### Example: Resistance Measurement

$y = x + v$

- Y: Measured value
- X: Actual value
- V: Measurement noise

Squared error: $(Y-X)^2$

Minimize Squared Error

Measurement Model

- $y_1 = x + v_1$
- $y_2 = x + v_2$
- $y_3 = x + v_3$
- $y_4 = x + v_4$

Squared Error

- $e^2_1 = (y_1-x)^2 $
- $e^2_2 = (y_2-x)^2 $
- $e^2_3 = (y_3-x)^2 $
- $e^2_4 = (y_4-x)^2 $

Squared Error Criterion

$\hat x_{LS} = argmin_x(e^2_1+e^2_2+e^2_3+e^2_4) = \mathcal L_{LS}(x)$

Re-write to vector

$$
E =
\begin{bmatrix}
e_1 \\
e_2 \\
e_3 \\
e_4 \\
\end{bmatrix}
= y - Hx
=
\begin{bmatrix}
y_1 \\
y_2 \\
y_3 \\
y_4 \\
\end{bmatrix}
-
\begin{bmatrix}
1 \\
1 \\
1 \\
1 \\
\end{bmatrix}
x
$$

Where H is the Jacobian matrix
