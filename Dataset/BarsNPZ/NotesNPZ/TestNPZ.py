import cv2
import numpy as np

# Загружаем подготовленный датасет
data = np.load("barsnotes(Inverted, 12, 60, r0).npz", allow_pickle=True)
windows = data["images"]
answers = data["answers"]

print(len(answers))

notes = ["ля0", "сі0", "до1", "ре1", "мі1", "фа1", "соль1", "ля1", "сі1", "до2", "ре2", "мі2"]

for i in range(len(windows)):
    # Получаем окно
    window = np.array(windows[i][0], dtype=np.float32)  # shape (60, 5)
    window = (window * 255).astype(np.uint8)  # масштабируем обратно в 0-255

    # Расширяем для отображения (увеличиваем в ширину, чтобы было видно)
    scale = 10
    window_resized = cv2.resize(window, (window.shape[1]*scale, window.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

    # Получаем правильную метку
    label_index = np.argmax(answers[i])
    label_text = notes[label_index]

    # Делаем копию, чтобы добавить подпись снизу
    canvas = cv2.copyMakeBorder(window_resized, 0, 20, 0, 0, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(canvas, label_text, (0, window_resized.shape[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,), 1, cv2.LINE_AA)

    cv2.imshow("Window", canvas)

    key = cv2.waitKey(300)  # ждем нажатия любой клавиши
    if key == 27:  # ESC для выхода
        break

    print(answers[i])

cv2.destroyAllWindows()
