from DataLoader import load_dataset
from LedetsNet3 import nn
import random

x_train, y_train = load_dataset()

struct = [2000, 300, 300, 200, 200, 1]
x = []
y = []

windows = []
answers = []

window_width = 200
window_height = 10
center_radius = 1

for img_idx in range(len(x_train)):
    image_height = len(x_train[0][0])
    for row_start in range(image_height - window_height + 1):
        window = []
        for row in range(row_start, row_start + window_height):
            window.append(x_train[img_idx][0][row][0:window_width])
        windows.append([window])

        center_pos = row_start + window_height // 2

        if center_pos - center_radius < 0 or center_pos + center_radius >= len(y_train[img_idx]):
            continue

        label_slice = y_train[img_idx][
                      center_pos - center_radius:
                      center_pos + center_radius + 1
                      ]

        label = 1 if max(label_slice) == 1 else 0
        answers.append([label])

print("Before balance:", len(windows),
      "pos=", sum(a[0] for a in answers),
      "neg=", len(answers) - sum(a[0] for a in answers))

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

for xd, yd in dataset:
    xt = []
    for i in xd:
        for j in i:
            for g in j:
                xt.append(g)
    x.append(xt)
    y.append(yd)

print(x[100], len(x[100]))
print(y[0], len(y[0]))

net = nn(struct, correctAnswers=y, learningRate=0.01, inputLayers=x)
net.WeightsInit()
net.BiasesInit()
net.ErrorsInit()
net.StartLearning(10, path=r"D:\Folders\tempModels", name="pages_mlp7", sv=1000)