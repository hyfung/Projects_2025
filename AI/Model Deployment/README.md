# Machine Learning Model Deployment

There are numerous way to serve a model

- Serverless
  - AWS Lambda
- Monolithic or Container
  - Flask
  - FastAPI
- Integrated
  - Embed in application
  - Code to update or fetch model

## Serverless Function

- Use AWS Lambda (or similar cloud provider) to write a serverless function in Python
- Write `requirements.txt` and install packages
- Instantiate object of the detector
- Receives HTTP request and respond

> Write an AWS Lambda function to load a YOLOv5 model from AWS S3 and perform inference on request

```python
# Sample code here
```

## Container

Gives you more control

## Monolithic / Integrated

Instead of making the model a microservice, it is built in to the application itself
