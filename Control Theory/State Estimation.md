# State Estimation

## Central Limit Theorem

## Jacobian Matrix

- First order partial derivative
- Linearization of Non-linear function
- Linearization error
2
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

### Example

$y = x + v$

- Y: Measured value
- X: Actual value
- V: Measurement noise

Squared error: $(Y-X)^2$

Minimize Squared Error
