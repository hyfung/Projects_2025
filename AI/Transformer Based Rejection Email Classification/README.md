# Classifying Emails with Transformer and Encoders

## Raw Data

- A collection of emails from employers giving thanks for my time

## Preprocessing

- Load the files one by one
- Remove whitespaces
- Convert them into single sentence
- A list of sentences with labels 0
- I should gather more data with label 1

## Tokenization

```python
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Example data
texts = ["This is a positive example", "This is a negative example"]
labels = [1, 0]

# Create a Dataset object
dataset = Dataset.from_dict({"text": texts, "label": labels})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Splitting Dataset

```python
from sklearn.model_selection import train_test_split

train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
```

## Training Code

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.evaluate()

```

## Classification

```python
import torch

test_texts = ["This is a great product!", "I hate this."]
inputs = tokenizer(test_texts, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)  # Predicted labels
```

## Saving Models

```python
model.save_pretrained("./binary_classification_model")
tokenizer.save_pretrained("./binary_classification_model")
```
