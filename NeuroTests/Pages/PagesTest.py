import os
import cv2
from LedetsNet3 import cnn

#folder = r"D:\PythonProjects\NoteAnalize3\Dataset\PagesNPZ\Pages"
folder = r"D:\Folders\TestImages"

data = cnn.Data()
net = data.load(r"D:\Folders\Models\Pages\Pages.json")
#net = data.load(r"D:\Folders\tempModels\2page17, 0.21.json")

def smooth_preds(preds, max_gap=2):
    smoothed = preds.copy()
    gap_count = 0
    for i in range(len(preds)):
        if preds[i] == 0:
            gap_count += 1
        else:
            if 0 < gap_count <= max_gap:
                smoothed[i-gap_count:i] = [1]*gap_count
            gap_count = 0
    return smoothed

for filename in os.listdir(folder):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(folder, filename)
    img_copy = cv2.imread(img_path)
    h_original, w_original = img_copy.shape[:2]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (52, 152))
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = 255 - img

    img_list = img.astype(float).tolist()
    img_list = [[px / 255.0 for px in row] for row in img_list]

    preds = net.forward([img_list])
    preds = [1 if p >= 0.5 else 0 for p in preds]

    preds = smooth_preds(preds, max_gap=2)

    clPreds = []
    start = None
    for i, val in enumerate(preds):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            clPreds.append((start, i-1))
            start = None
    if start is not None:
        clPreds.append((start, len(preds)-1))

    differences = [b - a for (a, b) in clPreds]
    average_diff = sum(differences) / len(differences)

    for i in range(len(clPreds)):
        if clPreds[i][1] - clPreds[i][0] > average_diff * 2:
            cort = clPreds[i]
            av = (clPreds[i][0] + clPreds[i][1]) // 2
            del clPreds[i]
            clPreds.append((cort[0], av - 1))
            clPreds.append((av + 1, cort[1]))


    scale = h_original / 152
    for top, bottom in clPreds:
        top = int((top) * scale) - 25
        bottom = int((bottom) * scale) + 30
        alpha = 0.4
        overlay = img_copy.copy()

        cv2.rectangle(overlay, (0, top), (w_original, bottom), (0, 0, 255), -1)

        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

    cv2.imshow(filename, cv2.resize(img_copy, (500, 800)))
    print(clPreds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
