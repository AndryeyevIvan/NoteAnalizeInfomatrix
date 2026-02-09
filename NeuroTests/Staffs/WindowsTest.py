from LedetsNet2 import cnn
from DataLoader import load_dataset
import cv2
import numpy as np

x_train, y_train = load_dataset()
data = cnn.Data()
net = data.load(r"C:\Users\thund\Downloads\Windows.json")

window_width = 30
window_height = 30

for img_idx in range(len(x_train)):
    image_width = len(x_train[0][0][0])
    answers = [0] * 500
    for col_start in range(image_width - window_width + 1):
        window = []
        for row in range(window_height):
            window.append(x_train[img_idx][0][row][col_start:col_start + window_width])

        pred = net.forward([window])

        if pred[0] > 0.8:
            answers[col_start + 15] = 1
        else:
            answers[col_start + 15] = 0

        print(col_start, pred)

    img = np.array(x_train[img_idx][0])

    img_uint8 = (img * 255).astype("uint8")

    img_big = cv2.resize(img_uint8, (1000, 60), interpolation=cv2.INTER_NEAREST)

    for x in range(500):
        if answers[x] > 0:
            cv2.line(img_big, (x * 2, 0), (x * 2, 60), 255, 1)

    cv2.imwrite(f"C:/Users/thund/Downloads/{img_idx}.jpg", img_big)