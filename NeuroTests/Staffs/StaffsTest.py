import os
import cv2
import numpy as np
from LedetsNet2 import cnn

folder = r"C:\Users\thund\PycharmProjects\NoteAnalize3\Dataset\StaffsNPZ\Staffs"

data = cnn.Data()
net = data.load(r"C:\Users\thund\Downloads\Staffs.json")

for filename in os.listdir(folder):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(folder, filename)
    img_copy = cv2.imread(img_path)
    h_original, w_original = img_copy.shape[:2]

    # --- Подготовка изображения для сети ---
    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 20))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = 255 - img

    img_list = img.astype(float).tolist()
    img_list = [[px / 255.0 for px in row] for row in img_list]

    preds = net.forward([img_list])  # берем первую (и единственную) картинку

    print(preds)

    for i in range(0, len(preds), 2):
        if preds[i + 1] >= 0.2:

            # i в диапазоне 0..351 → масштабируем к ширине реальной картинки
            x = int(preds[i] * w_original)

            # Рисуем вертикальную линию
            cv2.line(
                img_copy,
                (x, 0),
                (x, h_original),
                (0, 0, 255),
                2
            )

    # --- Показ ---
    cv2.imshow(filename, cv2.resize(img_copy, (850, 100)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
