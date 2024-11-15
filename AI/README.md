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

### Problems

| Problem            | Definition                   | Cause | Solution |
| ------------------ | ---------------------------- | ----- | -------- |
| Vanishing gradient | Model fails to learn feature |       |          |
| Data drift         |                              |       |          |
| Overfitting        | Model fails to generalize    |       |          |
|                    |                              |       |          |
|                    |                              |       |          |
