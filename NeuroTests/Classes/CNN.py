from LedetsNet2 import cnn
from DataLoader import load_dataset
import os
import random

def main():
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
    avg_count = 50
    print("\nAverage count per class:", avg_count)

    balanced_indices = []

    for cls, idxs in class_indices.items():
        bi = []
        if len(idxs) <= avg_count:
            # balanced_indices += idxs
            while len(bi) < avg_count:
                bi += random.sample(idxs, min(avg_count - len(bi), len(idxs)))
            balanced_indices += bi
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

    net = cnn.Model()

    net.add(cnn.Conv2d(numFilters=4, filterSize=3, inputDepth=1, stride=1, padding=1))
    net.add(cnn.ReLU())
    net.add(cnn.MaxPool2d(size=2, stride=2))

    net.add(cnn.Conv2d(numFilters=16, filterSize=3, inputDepth=4, stride=1, padding=1))
    net.add(cnn.ReLU())
    net.add(cnn.MaxPool2d(size=2, stride=2))

    net.add(cnn.Flatten())

    net.add(cnn.Linear(7904, 512))
    net.add(cnn.ReLU())

    net.add(cnn.Linear(512, 8))
    #net.add(cnn.Sigmoid())

    learning_rate = 0.02
    epochs = 20
    batch_size = 1
    data = cnn.Data()
    # net = data.load(r"C:\Users\ffedor\Downloads\Pages.json")

    for epoch in range(epochs):
        total_loss = 0
        lr = learning_rate * (0.995 ** epoch)
        print("LR:", lr)

        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            batch_preds = [net.forward(x) for x in batch_x]

            batch_loss = 0
            batch_dLoss = []
            for idx, (logits, y) in enumerate(zip(batch_preds, batch_y)):
                probs = cnn.softmax(logits)

                loss = cnn.crossEntropyLoss(probs, y)
                batch_loss += loss

                batch_dLoss.append(cnn.crossEntropyDerivative(probs, y))

                print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Sample {idx + 1}, Loss: {loss:.6f}")

            batch_loss /= len(batch_x)

            avg_dLoss = batch_dLoss[0]
            for d in batch_dLoss[1:]:
                avg_dLoss = [a+b for a,b in zip(avg_dLoss, d)]
            avg_dLoss = [a/len(batch_x) for a in avg_dLoss]

            # print("pred len:", len(batch_preds[0]))
            # print("y len:", len(batch_y[0]))
            # print("dLoss len:", len(avg_dLoss))

            net.backward(avg_dLoss)
            net.updateWeights(lr)

            total_loss += batch_loss

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / (len(x_train)/batch_size):.6f}")

        os.makedirs(r"D:\Folders\tempModels", exist_ok=True)
        data.save(model=net, path=r"D:\Folders\tempModels", fileName=f"classes_2{epoch}, {round(total_loss / (len(x_train)/batch_size),2)}")

if __name__ == "__main__":
    main()
