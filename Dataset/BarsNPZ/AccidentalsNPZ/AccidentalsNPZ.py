import os
import math
import cv2

folder = "Accidentals"
notes = ["", "д", "б", "бк"]

X = []
Y = []

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        base = os.path.splitext(filename)[0]
        txt_path = os.path.join(folder, f"{base}.txt")

        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 120), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img

        img_list = img.astype(float).tolist()
        img_list = [[pixel / 255.0 for pixel in row] for row in img_list]
        X.append([img_list])

        answers = [0] * 200

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "").replace("\r", "")
                content0 = content.split(":")[0]
                content1 = content.split(":")[1]
                cords = [float(x.replace(",", ".")) for x in content0.split(";") if x]
                nots = []
                for i in range(len(content1)):
                    if content1[i] == "б" and i != len(content1) - 1:
                        if content1[i + 1] == "к":
                            nots.append("бк")
                            continue
                    elif content1[i] == "к":
                        continue
                    elif len(content1) != 0:
                        nots.append(content1[i])

            for i in range(0, len(cords), 2):
                start = max(0, int(cords[i] * 200))
                end = min(200, int(cords[i + 1] * 200))
                for j in range(start, end):
                    answers[j] = notes.index(nots[i // 2])

        Y.append(answers)

print(len(X), len(X[0]), len(X[0][0]))

x_train = X
y_train = Y
windows = []
answers = []

window_width = 40
window_height = 120
center_radius = 0

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
        label = [0] * len(notes)
        label[max(label_slice)] = 1
        answers.append(label)
        print(label)

print("Before balance:", len(windows),
      "pos=", sum(1 - a[0] for a in answers),
      "neg=", len(answers) - sum(a[0] for a in answers))

import numpy as np
np.savez("barsaccidentals(Inverted, 40, 120, r0).npz", images=windows, answers=answers)

print(answers[40])