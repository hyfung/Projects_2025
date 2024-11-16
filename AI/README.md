# AI Projects

## Table of Content

## Quotes

> AI is only good at searching for pattern

> Gradient Descent is a greedy algorithm which looks for next local optimal

> An AI model is a graph which transform input into an output

> Training a model means tuning the parameter in each neuron to match input to expected output

## Scenario

### Model Deployment

There are numerous way to serve a model

- Serverless
  - AWS Lambda
- On-premise
  - Flask
  - FastAPI
- Integrated
  - Embed in application
  - Code to update or fetch model

### Continuous Integration

### Continuous Training

## Key Glossary Cheatsheet

### Training

| Name                   | Meaning                                      |
| ---------------------- | -------------------------------------------- |
| Loss Function          | Error for single data point                  |
| Cost Function          | Error for batch of data points               |
| Optimizer              | Decides how to update the weights            |
| Gradient Descent       | Greedy algorithm to reduce cost              |
| Batch Size             | How much data is used in an iteration        |
| Epoch                  | How many iteration to go through the dataset |
| mean Average Precision | Object detection metric with TP, TN, FP, FN  |

### Common Loss Functions

| Name                             | Measures                                                                          | Use Case                                 |
| -------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------- |
| Mean Squared Error               |                                                                                   | Regression                               |
| Mean Absolute Error              |                                                                                   | Regression                               |
| Binary Cross-Entropy             | How well the predicted probabilities match true labels                            | Binary Classification                    |
| Categorical Cross-Entropy        | How well the predicted probability distribution aligns with the true distribution | One Hot Multi Class Classification       |
| Sparse Categorical Cross-Entropy | Similar to categorical cross-entropy                                              | Integer Label Multi Class Classification |
|                                  |                                                                                   |                                          |
|                                  |                                                                                   |                                          |

### Common Optimizers

### Problems

| Problem            | Definition                   | Cause | Solution |
| ------------------ | ---------------------------- | ----- | -------- |
| Vanishing gradient | Model fails to learn feature |       |          |
| Data drift         |                              |       |          |
| Overfitting        | Model fails to generalize    |       |          |
|                    |                              |       |          |
|                    |                              |       |          |
