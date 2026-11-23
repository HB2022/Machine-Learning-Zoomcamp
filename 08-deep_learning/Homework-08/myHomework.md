# Homework 8: Neural Networks and Deep Learning

**Cohort:** ML Zoomcamp 2026

This file contains my submission for Homework 8. I followed the instructions exactly, using the fixed dataset, environment, reproducibility rules, and reference architecture. All code below was executed interactively in the terminal.

## Environment Setup

I installed the exact CPU wheels specified in `requirements-cpu.txt`:

```text
torch==2.9.0+cpu
torchvision==0.24.0+cpu
numpy==2.3.3
Pillow==11.3.0
```

Verification:

```text
torch: 2.9.0+cpu
torchvision: 0.24.0+cpu
numpy: 2.3.3
Pillow: 11.3.0
```

## Dataset

I downloaded the fixed dataset and verified the checksum:

```powershell
Invoke-WebRequest -Uri "[https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip](https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip)" -OutFile "data.zip"
(Get-FileHash .\data.zip -Algorithm SHA256).Hash.ToLower()
```

Checksum Output:

```text
9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67
```

Class counts verified:
- **Train:** 800 images
- **Held-out evaluation:** 201 images
- **Class mapping:** `{'curly': 0, 'straight': 1}`

## Reproducibility Rules and Data Loading

I set all seeds and configured PyTorch for deterministic execution before constructing any datasets, loaders, or models.

```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import statistics
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu")

baseline_transform = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = Path("data/train")
eval_dir = Path("data/test")

train_dataset = datasets.ImageFolder(train_dir, transform=baseline_transform)
eval_dataset = datasets.ImageFolder(eval_dir, transform=baseline_transform)

loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0, generator=loader_generator)
eval_loader = DataLoader(eval_dataset, batch_size=20, shuffle=False, num_workers=0)
```

## Model Architecture and Q1/Q2

```python
class HairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 99 * 99, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.hidden(x))
        return self.output(x)

model = HairModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

total_params = sum(p.numel() for p in model.parameters())
```

### Question 1
**Answer:** `nn.BCEWithLogitsLoss()`  
The model returns raw logits. `BCEWithLogitsLoss` combines a Sigmoid layer and `BCELoss` in one single class, which is more numerically stable.

### Question 2
**Answer:** `20073473`

## Baseline Training (10 Epochs) and Q3/Q4

```python
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

history_baseline = {"train_loss": [], "train_accuracy": [], "evaluation_loss": [], "evaluation_accuracy": []}

for epoch in range(10):
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * images.size(0)
        train_correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
        train_total += labels.size(0)
        
    epoch_train_loss = train_loss_sum / train_total
    epoch_train_acc = train_correct / train_total
    epoch_eval_loss, epoch_eval_acc = evaluate(model, eval_loader, criterion, device)
    
    history_baseline["train_loss"].append(epoch_train_loss)
    history_baseline["train_accuracy"].append(epoch_train_acc)
    history_baseline["evaluation_loss"].append(epoch_eval_loss)
    history_baseline["evaluation_accuracy"].append(epoch_eval_acc)
    print(f"Epoch {epoch+1}/10 | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Eval Loss: {epoch_eval_loss:.4f} | Eval Acc: {epoch_eval_acc:.4f}")
```

```text
Epoch 1/10 | Train Loss: 0.6385 | Train Acc: 0.6450 | Eval Loss: 0.6010 | Eval Acc: 0.6567
Epoch 2/10 | Train Loss: 0.5441 | Train Acc: 0.7175 | Eval Loss: 0.6003 | Eval Acc: 0.6517
Epoch 3/10 | Train Loss: 0.4955 | Train Acc: 0.7550 | Eval Loss: 0.6405 | Eval Acc: 0.6219
Epoch 4/10 | Train Loss: 0.4330 | Train Acc: 0.7913 | Eval Loss: 0.6274 | Eval Acc: 0.6915
Epoch 5/10 | Train Loss: 0.4048 | Train Acc: 0.8087 | Eval Loss: 0.6293 | Eval Acc: 0.6617
Epoch 6/10 | Train Loss: 0.3953 | Train Acc: 0.8075 | Eval Loss: 0.6332 | Eval Acc: 0.7114
Epoch 7/10 | Train Loss: 0.3437 | Train Acc: 0.8363 | Eval Loss: 0.6895 | Eval Acc: 0.6716
Epoch 8/10 | Train Loss: 0.2806 | Train Acc: 0.8850 | Eval Loss: 0.6497 | Eval Acc: 0.7214
Epoch 9/10 | Train Loss: 0.2236 | Train Acc: 0.9100 | Eval Loss: 0.7114 | Eval Acc: 0.7164
Epoch 10/10 | Train Loss: 0.1584 | Train Acc: 0.9387 | Eval Loss: 1.0212 | Eval Acc: 0.6368
```

### Question 3
**Answer:** `0.81` (Median of `train_accuracy`)

### Question 4
**Answer:** `0.139` (`np.std(train_loss, ddof=0)`)

## Data Augmentation (10 Epochs) and Q5/Q6

I replaced only the training transform with the augmented pipeline. The model and optimizer were not recreated.

```python
train_transform_augmented = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomRotation(50, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmented_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform_augmented)
augmented_loader_generator = torch.Generator().manual_seed(SEED + 1)

augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=20, shuffle=True, num_workers=0, generator=augmented_loader_generator)

history_augmented = {"train_loss": [], "train_accuracy": [], "evaluation_loss": [], "evaluation_accuracy": []}

for epoch in range(10):
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0
    for images, labels in augmented_train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * images.size(0)
        train_correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
        train_total += labels.size(0)
        
    epoch_train_loss = train_loss_sum / train_total
    epoch_train_acc = train_correct / train_total
    epoch_eval_loss, epoch_eval_acc = evaluate(model, eval_loader, criterion, device)
    
    history_augmented["train_loss"].append(epoch_train_loss)
    history_augmented["train_accuracy"].append(epoch_train_acc)
    history_augmented["evaluation_loss"].append(epoch_eval_loss)
    history_augmented["evaluation_accuracy"].append(epoch_eval_acc)
    print(f"Epoch {epoch+1}/10 | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Eval Loss: {epoch_eval_loss:.4f} | Eval Acc: {epoch_eval_acc:.4f}")
```

```text
Epoch 1/10 | Train Loss: 0.6758 | Train Acc: 0.6050 | Eval Loss: 0.6552 | Eval Acc: 0.6716
Epoch 2/10 | Train Loss: 0.6129 | Train Acc: 0.6475 | Eval Loss: 0.6608 | Eval Acc: 0.6269
Epoch 3/10 | Train Loss: 0.5899 | Train Acc: 0.6800 | Eval Loss: 0.5787 | Eval Acc: 0.7114
Epoch 4/10 | Train Loss: 0.5923 | Train Acc: 0.6750 | Eval Loss: 0.5679 | Eval Acc: 0.7114
Epoch 5/10 | Train Loss: 0.5604 | Train Acc: 0.7137 | Eval Loss: 0.5680 | Eval Acc: 0.7214
Epoch 6/10 | Train Loss: 0.5567 | Train Acc: 0.7075 | Eval Loss: 0.5510 | Eval Acc: 0.7015
Epoch 7/10 | Train Loss: 0.5433 | Train Acc: 0.7000 | Eval Loss: 0.5291 | Eval Acc: 0.7413
Epoch 8/10 | Train Loss: 0.5317 | Train Acc: 0.7288 | Eval Loss: 0.5458 | Eval Acc: 0.7065
Epoch 9/10 | Train Loss: 0.5041 | Train Acc: 0.7562 | Eval Loss: 0.5557 | Eval Acc: 0.7214
Epoch 10/10 | Train Loss: 0.5398 | Train Acc: 0.7050 | Eval Loss: 0.5462 | Eval Acc: 0.7313
```

### Question 5
**Answer:** `0.576` (Mean of `evaluation_loss`)

### Question 6
**Answer:** `0.72` (Mean of last 5 `evaluation_accuracy`)

## Saving History Files

```python
with open("history_baseline.json", "w") as f:
    json.dump(history_baseline, f, indent=4)

with open("history_augmented.json", "w") as f:
    json.dump(history_augmented, f, indent=4)
```

## Summary of Answers

| Question | Answer |
| :--- | :--- |
| **Q1. Loss Function** | `nn.BCEWithLogitsLoss()` |
| **Q2. Parameter Count** | `20073473` |
| **Q3. Baseline Accuracy** | `0.81` |
| **Q4. Baseline Loss (Std)** | `0.139` |
| **Q5. Augmented Eval Loss** | `0.576` |
| **Q6. Augmented Eval Acc** | `0.72` |
