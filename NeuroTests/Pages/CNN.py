from LedetsNet3 import cnn
from DataLoader import load_dataset
import os

def main():
    x_train, y_train = load_dataset()
    x_test, y_test = x_train, y_train

    net = cnn.Model()

    net.add(cnn.Conv2d(4, 3, inputDepth=1, padding=1))
    net.add(cnn.ReLU())

    net.add(cnn.Conv2d(4, 3, inputDepth=4, padding=1))
    net.add(cnn.ReLU())

    net.add(cnn.MaxPool2d(size=(1, 2), stride=(1, 2)))  # 52 → 26
    net.add(cnn.MaxPool2d(size=(1, 2), stride=(1, 2)))
    net.add(cnn.MaxPool2d(size=(1, 13), stride=(1, 13)))

    net.add(cnn.Conv2d(1, 1, inputDepth=4))

    net.add(cnn.Flatten())
    net.add(cnn.Sigmoid())

    learning_rate = 0.05
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

            # print("pred len:", len(batch_preds[0]))
            # print("y len:", len(batch_y[0]))
            # print("dLoss len:", len(avg_dLoss))

            net.backward(avg_dLoss)
            net.updateWeights(lr)

            total_loss += batch_loss

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / (len(x_train)/batch_size):.6f}")

        os.makedirs(r"D:\Folders\tempModels", exist_ok=True)
        data.save(model=net, path=r"D:\Folders\tempModels", fileName=f"4page1{epoch}, {round(total_loss / (len(x_train)/batch_size),2)}")

    total_acc = 0
    for x, y in zip(x_test, y_test):
        preds = net.forward(x)
        preds_bin = [1 if p >= 0.5 else 0 for p in preds]
        acc = sum(p == t for p, t in zip(preds_bin, y)) / len(y)
        total_acc += acc
    print(f"Test pixel accuracy: {total_acc / len(x_test):.4f}")

if __name__ == "__main__":
    main()
