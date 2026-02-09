from LedetsNet3 import nn
from DataLoader import load_dataset
import cv2
import numpy as np

x_train, y_train = load_dataset()
net = nn()
net.load(path=r"C:\Users\Иван\Downloads\groups_emp_mlp1_epoch_8_loss_0.01.json")

window_width = 10
window_height = 60

answers = []

for img_idx in range(len(x_train)):
    image_width = len(x_train[0][0][0])
    answers = [0] * 200
    for col_start in range(image_width - window_width + 1):
        window = []
        for row in range(window_height):
            for x in x_train[img_idx][0][row][col_start:col_start + window_width]:
                window.append(x)

        net.struct[0] = window
        net.ForwardPropagation()
        pred = net.struct[-1]

        answers[col_start + 5] = pred[0]

        # if pred[0] > 0.8:
        #     answers[col_start + 15] = 1
        # else:
        #     answers[col_start + 15] = 0

        print(col_start, pred)

    c = ((sum(answers) / len(answers)) + max(answers)) / 2

    print(c)

    for i in range(len(answers)):
        if answers[i] >= c:
            answers[i] = 1
        else:
            answers[i] = 0

    d = []
    t = [0, 0]
    c = 0

    for i in range(len(answers)):
        if t[0] == 0 and answers[i] == 1:
            t[0] = i

        if t[0] != 0 and answers[i] == 0:
            if c > 2:
                c = 0
                t[1] = i - 3
                d.append(t)
                t = [0, 0]
            else:
                c += 1

    if t[0] != 0:
        t[1] = len(answers)
        d.append(t)

    img = np.array(x_train[img_idx][0])

    img_uint8 = (img * 255).astype("uint8")

    img_big = cv2.resize(img_uint8, (200, 60), interpolation=cv2.INTER_NEAREST)

    st = 0
    mt = 0

    for x1, x2 in d:
        l = x2 - x1
        st += l
        if l > mt:
            mt = l

    st = st / len(d)

    # avet = ((st / len(d)) + mt) / 2 - 3
    avet = 2

    centers = []

    for x1, x2 in d:
        if x2 - x1 > avet:
            center = int((x1 + x2) // 2)
            centers.append(center)

    centers = sorted(centers)
    cuts = [0] + centers + [img_big.shape[1]]

    for i in range(len(cuts) - 1):
        x_start = cuts[i]
        x_end = cuts[i + 1]

        if x_end - x_start < 2:
            continue

        piece = img_big[:, x_start:x_end]

        cv2.imwrite(f"D:/Folders/DFF/{img_idx}_piece_{i}.jpg", 255 - piece)

    # for x in range(200):
    #     if answers[x] > 0.6:
    #         cv2.line(img_big, (x, 0), (x, 60), 255, 1)

    #cv2.imwrite(f"D:/Folders/TestImages2/{img_idx}_mlp.jpg", img_big)