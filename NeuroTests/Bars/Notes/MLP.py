from DataLoader import load_dataset
from LedetsNet3 import nn
import random

x_train, y_train = load_dataset()

struct = [720, 50, 50, 50, 50, 13]
x = []
y = []

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
avg_count = 300
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

for xd in range(len(x_train)):
    xt = []
    for i in x_train[xd]:
        for j in i:
            for g in j:
                xt.append(g)
    x.append(xt)

y = y_train

print(x[100], len(x[100]))
print(y[0], len(y[0]))

net = nn(struct, correctAnswers=y, learningRate=0.01, inputLayers=x)
net.WeightsInit()
net.BiasesInit()
net.ErrorsInit()
net.StartLearning(80, path=r"C:\Users\Иван\Downloads", name="notes_mlp2_update")