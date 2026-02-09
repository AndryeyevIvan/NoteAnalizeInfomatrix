import os
import cv2

folder = "Bars"

X = []
y = []

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        base = os.path.splitext(filename)[0]
        txt_path = os.path.join(folder, f"{base}.txt")

        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 60), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img

        img_list = img.astype(float).tolist()
        img_list = [[pixel / 255.0 for pixel in row] for row in img_list]
        X.append([img_list])

        answers = [1.0] * 200

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "").replace("\r", "")
                content = content[:-1]
                cords = [float(x.replace(",", ".")) for x in content.split(";") if x]

            for i in range(0, len(cords), 2):
                start = max(0, int(cords[i] * 200))
                end = min(200, int(cords[i + 1] * 200))
                for j in range(start, end):
                    answers[j] = 0.0

        y.append(answers)
        print(answers)

print(len(X), len(X[0]), len(X[0][0]))

import numpy as np
np.savez("barsgroups(Inverted, 200, 60, emp).npz", images=X, answers=y)

print("Датасет сохранён как pages200(Smoll).npz")
print("Картинок:", len(X))
