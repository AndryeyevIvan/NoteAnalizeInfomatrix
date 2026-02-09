from LedetsNet2 import cnn
from DataLoader import load_dataset
import os
import math
import random

# ===== BCE LOSS =====
def bce_loss(pred, target):
    eps = 1e-9
    p = min(max(pred[0], eps), 1 - eps)
    t = float(target[0])
    return -(t * math.log(p) + (1 - t) * math.log(1 - p))

# ===== BCE dL/dp =====
def bce_loss_derivative(pred, target):
    eps = 1e-9
    p = min(max(pred[0], eps), 1 - eps)
    t = float(target[0])
    return [-(t / p) + (1 - t) / (1 - p)]    # dL/dp

# ===== LOAD DATA =====
x_train, y_train = load_dataset()

# ===== CREATE WINDOWS (30 × 30) =====
windows = []
answers = []

window_width = 30
window_height = 30
center_radius = 3

for img_idx in range(len(x_train)):
    image_width = len(x_train[0][0][0])
    for col_start in range(image_width - window_width + 1):
        window = []
        for row in range(window_height):
            window.append(x_train[img_idx][0][row][col_start:col_start + window_width])
        windows.append([window])

        center_col = window_width // 2
        start_check = center_col - center_radius
        end_check = center_col + center_radius + 1
        label_slice = y_train[img_idx][col_start + start_check: col_start + end_check]
        label = 1 if max(label_slice) == 1 else 0
        answers.append([label])

print("Before balance:", len(windows),
      "pos=", sum(a[0] for a in answers),
      "neg=", len(answers) - sum(a[0] for a in answers))

# ===== BALANCE CLASSES =====
pos_idx = [i for i, a in enumerate(answers) if a[0] == 1]
neg_idx = [i for i, a in enumerate(answers) if a[0] == 0]
min_class = min(len(pos_idx), len(neg_idx))

pos_idx = random.sample(pos_idx, min_class)
neg_idx = random.sample(neg_idx, min_class)

dataset = [(windows[i], answers[i]) for i in pos_idx] + \
          [(windows[i], answers[i]) for i in neg_idx]

random.shuffle(dataset)

print("After balance:", len(dataset),
      "pos=", sum(y[0] for x, y in dataset),
      "neg=", len(dataset) - sum(y[0] for x, y in dataset))

# ===== BUILD MODEL =====
net = cnn.Model()

net.add(cnn.Conv2d(16, 3, 1, stride=1, padding=1))
net.add(cnn.ReLU())
net.add(cnn.MaxPool2d(2, 2))

net.add(cnn.Conv2d(32, 3, 16, stride=1, padding=1))
net.add(cnn.ReLU())

net.add(cnn.Flatten())
net.add(cnn.Linear(7200, 128))
net.add(cnn.ReLU())
net.add(cnn.Linear(128, 1))
net.add(cnn.Sigmoid())

# ===== TRAIN =====
learning_rate = 0.01
epochs = 20
data = cnn.Data()

for epoch in range(epochs):
    total_loss = 0.0
    lr = learning_rate * (0.995 ** epoch)

    random.shuffle(dataset)
    print("\nEpoch", epoch + 1, "LR =", lr)

    for i, (x, y) in enumerate(dataset):

        # forward
        pred = net.forward(x)
        loss = bce_loss(pred, y)
        total_loss += loss

        # backward
        grad = bce_loss_derivative(pred, y)
        net.backward(grad)

        # update
        net.updateWeights(lr)

        if i % 1 == 0:
            print(f"  Sample {i}: loss = {loss:.6f}, y:{y}")

        if i % 1000 == 0:
            os.makedirs("C:/Users/thund/Downloads", exist_ok=True)
            data.save(net, "C:/Users/thund/Downloads", f"model2_epoch_{i}")

    print("Epoch avg loss:", total_loss / len(dataset))

    os.makedirs("C:/Users/thund/Downloads", exist_ok=True)
    data.save(net, "C:/Users/thund/Downloads", f"model2_epoch_{epoch}")

# ===== TEST =====
correct = 0
for x, y in dataset:
    p = net.forward(x)[0]
    pred = 1 if p >= 0.5 else 0
    if pred == y[0]:
        correct += 1

print("\nAccuracy:", correct / len(dataset))
