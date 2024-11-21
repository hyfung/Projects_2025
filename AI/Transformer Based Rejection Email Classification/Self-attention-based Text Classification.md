# Using Self Attention Mechanism to Classify Emails

## Importing Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
```

## Prepare Dataset

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
```

## Self Attention Layer

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = embed_size ** 0.5

    def forward(self, x, mask=None):
        # Compute queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute the output
        output = torch.bmm(attention_weights, V)
        return output, attention_weights
```

## Classification Model

```python
class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = SelfAttention(embed_size)
        self.fc = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embed_size)
        attention_output, _ = self.attention(embedded)  # Apply self-attention
        pooled_output = torch.mean(attention_output, dim=1)  # Pooling across sequence
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)  # Final linear layer
        return logits
```

## Prepare the Data

```python
# Hyperparameters
vocab_size = 30000  # Tokenizer vocab size
embed_size = 128    # Embedding size
num_classes = 2     # Binary classification
max_len = 128       # Max sequence length
batch_size = 32
epochs = 5
learning_rate = 1e-4

# Example data

# Load rejection data from folder
texts_rejection = []
for filename in [os.path.join('data/rejection', filename) for filename in os.listdir('data/rejection')]:
    with open(filename, 'r') as f:
        texts_rejection += f.readlines()

texts_rejection = [text.strip('\n').strip(' ') for text in texts_rejection]
texts_rejection = [text for text in texts_rejection if text != '']
labels_rejection = [1] * len(texts_rejection)

# Load nonrejection data from folder
texts_nonrejection = []
for filename in [os.path.join('data/rejection', filename) for filename in os.listdir('data/rejection')]:
    with open(filename, 'r') as f:
        texts_nonrejection += f.readlines()

texts_nonrejection = [text.strip('\n').strip(' ') for text in texts_nonrejection]
texts_nonrejection = [text for text in texts_nonrejection if text != '']
labels_nonrejection = [1] * len(texts_nonrejection)

# Create dataset
texts = texts_rejection + texts_nonrejection
labels = labels_rejection + labels_nonrejection

# Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dataset and DataLoader
dataset = TextDataset(texts, labels, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = SelfAttentionClassifier(vocab_size, embed_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

## Training the Model

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
```

## Evaluating the Model

```python
def predict(text, model, tokenizer, max_len):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).item()

test_text = "I really enjoyed this movie!"
prediction = predict(test_text, model, tokenizer, max_len)
print(f"Prediction: {prediction}")
```
