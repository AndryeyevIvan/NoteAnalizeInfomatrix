import cv2
import numpy as np
from LedetsNet2 import cnn
from DataLoader import load_dataset

def main():
    # --- Загружаем данные ---
    x_train, y_train = load_dataset()
    x_test, y_test = x_train, y_train

    data = cnn.Data()
    net = data.load(r"C:\Users\ffedor\Desktop\Models\Pages\Pages.json")

    # === Тестирование ===
    # total_acc = 0
    # for i, (x, y) in enumerate(zip(x_test, y_test), 1):
    #     preds = net.forward(x)
    #     preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    #     acc = sum(p == t for p, t in zip(preds_bin, y)) / len(y)
    #     total_acc += acc
    #
    # print(f"\n=== Final average pixel accuracy: {total_acc / len(x_test):.4f} ===\n")

    # === Визуализация ===
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        preds = net.forward(x)

        # Преобразуем x в нормальное изображение
        img = np.array(x[0], dtype=np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w, _ = img.shape  # должно быть (200, ширина, 3)

        # Рисуем реальные линии
        for idx, t in enumerate(y):
            if t > 0.5:
                color = (0, 255, 0)  # зелёная линия — ground truth
                overlay = img.copy()
                cv2.line(overlay, (0, idx), (w, idx), color, 1)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Рисуем предсказанные линии
        for idx, p in enumerate(preds):
            if p > 0.5:
                color = (0, 0, 255)  # красная линия — предсказание
                overlay = img.copy()
                cv2.line(overlay, (0, idx), (w, idx), color, 1)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Показываем результат
        cv2.imshow(f"Sample {i}", cv2.resize(img, (500, 1000)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
