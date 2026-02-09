from LedetsNet2 import cnn
from DataLoader import load_dataset
import os

def main():
    x_train, y_train = load_dataset()
    x_test, y_test = x_train, y_train

    net = cnn.Model()

    # Вход: (1, 152, 52)

    net.add(cnn.Conv2d(numFilters=8, filterSize=3, inputDepth=1, stride=1, padding=1))  # → (8,152,52)
    net.add(cnn.ReLU())
    net.add(cnn.MaxPool2d(size=2, stride=2))  # → (8,76,26)

    net.add(cnn.Conv2d(numFilters=16, filterSize=3, inputDepth=8, stride=1, padding=1))  # → (16,76,26)
    net.add(cnn.ReLU())
    net.add(cnn.MaxPool2d(size=2, stride=2))  # → (16,38,13)

    net.add(cnn.Flatten())  # → 16*38*13 = 7904

    net.add(cnn.Linear(7904, 256))
    net.add(cnn.ReLU())

    net.add(cnn.Linear(256, 152))  # выход по вертикали
    net.add(cnn.Sigmoid())

    learning_rate = 0.15
    epochs = 300
    batch_size = 4
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
            for idx, (pred, y) in enumerate(zip(batch_preds, batch_y)):
                loss = cnn.binaryCrossEntropy(pred, y)
                batch_loss += loss
                batch_dLoss.append(cnn.binaryCrossEntropyDerivative(pred, y))

                print(f"Epoch {epoch+1}, Batch {i//batch_size + 1}, Sample {idx+1}, Loss: {loss:.6f}")

            batch_loss /= len(batch_x)

            avg_dLoss = batch_dLoss[0]
            for d in batch_dLoss[1:]:
                avg_dLoss = [a+b for a,b in zip(avg_dLoss, d)]
            avg_dLoss = [a/len(batch_x) for a in avg_dLoss]

            net.backward(avg_dLoss)
            net.updateWeights(lr)

            total_loss += batch_loss

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / (len(x_train)/batch_size):.6f}")

        os.makedirs(r"C:\Users\ffedor\Downloads", exist_ok=True)
        data.save(model=net, path=r"C:\Users\ffedor\Downloads", fileName=f"pagesB{epoch}, {round(total_loss / (len(x_train)/batch_size),2)}")

    total_acc = 0
    for x, y in zip(x_test, y_test):
        preds = net.forward(x)
        preds_bin = [1 if p >= 0.5 else 0 for p in preds]
        acc = sum(p == t for p, t in zip(preds_bin, y)) / len(y)
        total_acc += acc
    print(f"Test pixel accuracy: {total_acc / len(x_test):.4f}")

if __name__ == "__main__":
    main()
