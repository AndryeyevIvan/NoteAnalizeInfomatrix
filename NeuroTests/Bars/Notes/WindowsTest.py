from LedetsNet2 import cnn
from DataLoader2 import load_dataset
import cv2
import numpy as np

x_train, y_train = load_dataset()
data = cnn.Data()
net = data.load(r"C:\Users\thund\Downloads\Windows5.json")
# notes = ["", "ля0", "сі0", "до1", "ре1", "мі1", "фа1", "соль1", "ля1", "сі1", "до2", "ре2", "мі2"]
notes = ["-", "la0", "si0", "do1", "re1", "mi1", "fa1", "sol1", "la1", "si1", "do2", "re2", "mi2"]

window_width = 12
window_height = 60

for img_idx in range(len(x_train)):
    image_width = len(x_train[0][0][0])

    text = ""

    for col_start in range(image_width - window_width + 1):
        window = []
        for row in range(window_height):
            window.append(x_train[img_idx][0][row][col_start:col_start + window_width])

        pred = net.forward([window])

        if max(pred) > 1:
            print(notes[pred.index(max(pred))])
            text = notes[pred.index(max(pred))]
        else:
            print("-")
            text = "-"

    img = np.array(x_train[img_idx][0])

    img_uint8 = (img * 255).astype("uint8")

    img_big = cv2.resize(img_uint8, (80, 480), interpolation=cv2.INTER_NEAREST)

    img_big = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_big, text, (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow(f"img", img_big)
    cv2.waitKey(1)