from LedetsNet3 import nn
import os
import cv2

folder = r"D:\PythonProjects\NoteAnalize3\Dataset\BarsNPZ\NotesNPZ\Bars"
notes = ["", "ля0", "сі0", "до1", "ре1", "мі1", "фа1", "соль1", "ля1", "сі1", "до2", "ре2", "мі2"]

net = nn()
net.load(path=r"C:\Users\Иван\Downloads\notes_mlp2_update_epoch_79_loss_0.01.json")

files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

X = []

for filename in files:
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        base = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 60), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img

        img_list = img.astype(float).tolist()
        img_list = [[pixel / 255.0 for pixel in row] for row in img_list]
        X.append([img_list])

print(len(X), len(X[0]), len(X[0][0]))

x_train = X

answers = []

window_width = 12
window_height = 60

for img_idx in range(len(x_train)):
    image_width = len(x_train[0][0][0])
    ans = []
    for col_start in range(image_width - window_width + 1):
        wind = []
        for row in range(window_height):
            wind.append(x_train[img_idx][0][row][col_start:col_start + window_width])

        window = []
        for row in wind:
            for px in row:
                window.append(px)

        net.struct[0] = window
        net.ForwardPropagation()

        pred = net.struct[-1]

        ans.append(pred.copy())

    T = len(ans)
    C = len(ans[0])

    def smooth_probs(probs, k):
        half = k // 2
        smoothed = []

        for t in range(T):
            row = []
            for c in range(C):
                s = 0.0
                cnt = 0

                for dt in range(-half, half + 1):
                    tt = t + dt
                    if 0 <= tt < T:
                        s += probs[tt][c]
                        cnt += 1

                row.append(s / cnt)
            smoothed.append(row)

        return smoothed

    smoothed = smooth_probs(ans, k=3)

    threshold = 0.3
    minLen = 5

    events = []

    for note in range(1, C):
        start = None

        for t in range(T):
            if smoothed[t][note] > threshold:
                if start is None:
                    start = t
            else:
                if start is not None:
                    if t - start >= minLen:
                        events.append((note, start, t))
                    start = None

        if start is not None and T - start >= minLen:
            events.append((note, start, T))

    filteredEvents = []

    for note, s, e in events:
        score = 0.0
        for t in range(s, e):
            score += smoothed[t][note]

        filteredEvents.append((note, s, e, score))

    filteredEvents.sort(key=lambda x: x[1])

    answers.append(filteredEvents)

for i, filename in enumerate(files):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)

        print(answers[i])

        answ = list(map(lambda x: notes[x[0]], answers[i]))
        print(answ)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 60), interpolation=cv2.INTER_AREA)
        cv2.imshow("bar", img)
        cv2.waitKey(0)