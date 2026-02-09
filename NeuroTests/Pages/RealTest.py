import os
import cv2
from LedetsNet3 import nn
from DataLoader import load_dataset
import cv2
import numpy as np

folder = r"D:\Folders\TestIm"

X = []

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        img = cv2.imread(img_path)
        coef = img.shape[1] / 100
        print(coef)
        img = cv2.resize(img, (int(img.shape[1] / coef), int(img.shape[0] / coef)), interpolation=cv2.INTER_AREA)
        print(img.shape)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img

        img_list = img.astype(float).tolist()
        img_list = [[pixel / 255.0 for pixel in row] for row in img_list]
        X.append([img_list])

x_train = X

net = nn()
net.load(path=r"C:\Users\Иван\Downloads\pages_mlp6_sample_7000.json")

window_width = 100
window_height = 5

for img_idx in range(len(x_train)):
    img = x_train[img_idx][0]
    image_height = len(img)
    image_width = len(img[0])

    answers = []

    # === СКОЛЬЖЕНИЕ ПО ВЕРТИКАЛИ (КАК В ОБУЧЕНИИ) ===
    for row_start in range(image_height - window_height + 1):
        window = []

        for row in range(row_start, row_start + window_height):
            for px in img[row][0:window_width]:
                window.append(px)

        # подаём ровно 260 входов
        net.struct[0] = window
        net.ForwardPropagation()

        pred = net.struct[-1][0]
        answers.append(pred)

        print(f"row={row_start} pred={pred:.3f}")

    # === АДАПТИВНЫЙ ПОРОГ (КАК У ТЕБЯ) ===
    # c = ((sum(answers) / len(answers)) + max(answers)) / 2
    c = 0.8

    bin_answers = [1 if a >= c else 0 for a in answers]

    # === ПОИСК НЕПРЕРЫВНЫХ УЧАСТКОВ ===
    segments = []
    start = None
    gap = 0

    for i, v in enumerate(bin_answers):
        if v == 1 and start is None:
            start = i
            gap = 0

        elif v == 0 and start is not None:
            gap += 1
            if gap > 2:
                end = i - gap
                segments.append((start, end))
                start = None

    if start is not None:
        segments.append((start, len(bin_answers)))

    # === ВИЗУАЛИЗАЦИЯ ===
    img_uint8 = (np.array(img) * 255).astype("uint8")
    img_big = cv2.resize(img_uint8, (image_width * 2, image_height * 2),
                         interpolation=cv2.INTER_NEAREST)

    # for y1, y2 in segments:
    #     center = (y1 + y2) // 2
    #     cv2.line(
    #         img_big,
    #         (0, center * 2),
    #         (image_width * 2, center * 2),
    #         255,
    #         1
    #     )

    for y1, y2 in segments:
        for i in range(y1, y2):
            cv2.line(
                img_big,
                (0, i * 2),
                (image_width * 2, i * 2),
                255,
                1
            )

    cv2.imwrite(f"D:/Folders/TestImages/{img_idx}_mlp.jpg", img_big)