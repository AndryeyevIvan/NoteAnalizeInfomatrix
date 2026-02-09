from LedetsNet3 import nn
from DataLoader import load_dataset
import cv2
import numpy as np

x_train, y_train = load_dataset()

net = nn()
net.load(path=r"C:\Users\Иван\Downloads\notes_mlp2_epoch_39_loss_0.03.json")

notes = ["-", "la0", "si0", "do1", "re1", "mi1", "fa1", "sol1",
         "la1", "si1", "do2", "re2", "mi2"]


allPreds = []
allImages = []

for img_idx in range(len(x_train)):
    window = []
    for row in x_train[img_idx][0]:
        for px in row:
            window.append(px)

    net.struct[0] = window
    net.ForwardPropagation()

    pred = net.struct[-1]
    allPreds.append(pred.copy())
    allImages.append(x_train[img_idx][0])

T = len(allPreds)
C = len(allPreds[0])

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

smoothed = smooth_probs(allPreds, k=7)

threshold = 0.4
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

print(filteredEvents)

for note, start, end, score in filteredEvents:
    for t in range(start, end):
        img = np.array(allImages[t])
        img_uint8 = (img * 255).astype("uint8")

        img_big = cv2.resize(
            img_uint8, (80, 480),
            interpolation=cv2.INTER_NEAREST
        )
        img_big = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            img_big,
            notes[note],
            (5, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow("Filtered output", img_big)
        cv2.waitKey(100)

cv2.destroyAllWindows()
