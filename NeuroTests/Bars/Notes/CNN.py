from LedetsNet2 import cnn
import numpy as np
import math
import random
from DataLoader2 import load_dataset
import os

x_train, y_train = load_dataset()
print("Loaded dataset:", len(x_train), "samples")

class_indices = {}
for i, y in enumerate(y_train):
    cls = y.index(1)
    if cls not in class_indices:
        class_indices[cls] = []
    class_indices[cls].append(i)

counts = {cls: len(idxs) for cls, idxs in class_indices.items()}

print("\nBefore balance:")
for cls in sorted(counts):
    print(f"  class {cls}: {counts[cls]}")

# avg_count = int(sum(counts.values()) / len(counts)) - 500
# avg_count = min(counts.values())
avg_count = 60
print("\nAverage count per class:", avg_count)

balanced_indices = []

for cls, idxs in class_indices.items():
    if len(idxs) <= avg_count:
        balanced_indices += idxs
    else:
        balanced_indices += random.sample(idxs, avg_count)

random.shuffle(balanced_indices)

x_bal = [x_train[i] for i in balanced_indices]
y_bal = [y_train[i] for i in balanced_indices]

x_train = x_bal
y_train = y_bal

print("\nAfter balance:", len(x_bal), "samples")

new_counts = {}
for y in y_bal:
    cls = y.index(1)
    new_counts[cls] = new_counts.get(cls, 0) + 1

for cls in sorted(new_counts):
    print(f"  class {cls}: {new_counts[cls]}")

def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]

def cross_entropy(pred, target):
    eps = 1e-9
    return -sum(target[i] * math.log(pred[i] + eps) for i in range(len(pred)))

def cross_entropy_derivative(pred, target):
    return [pred[i] - target[i] for i in range(len(pred))]

net = cnn.Model()

# 1 блок
net.add(cnn.Conv2d(numFilters=16, filterSize=3, inputDepth=1, stride=1, padding=1))
net.add(cnn.ReLU())

net.add(cnn.MaxPool2d(size=2, stride=2))   # 60x10 → 30x6

# 2 блок
net.add(cnn.Conv2d(numFilters=32, filterSize=3, inputDepth=16, stride=1, padding=1))
net.add(cnn.ReLU())

net.add(cnn.MaxPool2d(size=2, stride=2))   # 30x5 → 15x3

# 3 блок (минимальный)
net.add(cnn.Conv2d(numFilters=32, filterSize=3, inputDepth=32, stride=1, padding=1))
net.add(cnn.ReLU())

# Размер: 15 x 2 x 32 = 960
net.add(cnn.Flatten())

# Dense
net.add(cnn.Linear(1440, 128))
net.add(cnn.ReLU())

net.add(cnn.Linear(128, 13))   # выход

data = cnn.Data()

learning_rate = 0.02
epochs = 20

for epoch in range(epochs):
    total_loss = 0.0

    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    x_train, y_train = zip(*combined)

    print(f"\nEpoch {epoch+1}/{epochs}")

    for i, (x, y) in enumerate(zip(x_train, y_train)):
        logits = net.forward(x)
        pred = softmax(logits)
        loss = cross_entropy(pred, y)

        grad = cross_entropy_derivative(pred, y)
        net.backward(grad)
        net.updateWeights(learning_rate)

        total_loss += loss

        if i % 1 == 0:
            print(f"  sample {i}: loss={loss:.5f}, y:{y}")

        # if i % 1000 == 0:
        #     data.save(net, "C:/Users/thund/Downloads", f"bars_model_epoch_{epoch}_sample_{i}")

    print("Epoch avg loss:", total_loss / len(x_train))

    data.save(net, "C:/Users/thund/Downloads", f"bars_model2_epoch_{epoch}")

correct = 0
total = len(x_train)

for x, y in zip(x_train, y_train):
    logits = net.forward(x)
    pred = softmax(logits)

    pred_class = pred.index(max(pred))
    true_class = y.index(1)

    if pred_class == true_class:
        correct += 1

print("\nAccuracy:", correct / total)
