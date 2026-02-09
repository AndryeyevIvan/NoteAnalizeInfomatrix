import os
import cv2

folder = "Classes"
notes = ["", "4", "28", "2", "ск", "4/4", "4п", "2/4"]

X = []
y = []

def resize_with_padding_white(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=255)

    return padded

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        base = os.path.splitext(filename)[0]
        txt_path = os.path.join(folder, f"{base}.txt")

        answers = [0.0] * 8

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "").replace("\r", "")

                if content == "брак":
                    continue

                answers[notes.index(content)] = 1.0

        y.append(answers)

        print(answers)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = resize_with_padding_white(img, 152, 52)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img

        img_list = img.astype(float).tolist()
        img_list = [[pixel / 255.0 for pixel in row] for row in img_list]
        X.append([img_list])

print(len(X), len(X[0]), len(X[0][0]))

import numpy as np
np.savez("classes(152, 52, 2v).npz", images=X, answers=y)

print("Датасет сохранён как pages200(Smoll).npz")
print("Картинок:", len(X))
