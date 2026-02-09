import os
import cv2
from LedetsNet3 import cnn

folder = r"D:\PythonProjects\NoteAnalize3\Dataset\ClassesNPZ\Classes"

data = cnn.Data()
net = data.load(r"D:\Folders\tempModels\classes_21, 0.22.json")

classes = ["-", "4", "28", "2", "sk", "4/4", "4p", "2/4"]

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
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(folder, filename)
    img_copy = cv2.imread(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = resize_with_padding_white(img, 152, 52)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = 255 - img

    img_list = img.astype(float).tolist()
    img_list = [[px / 255.0 for px in row] for row in img_list]

    logits = net.forward([img_list])

    #probs = cnn.softmax(logits)
    probs = logits

    idx = probs.index(max(probs))
    pred = classes[idx]

    print(probs, idx, pred)

    cv2.putText(img_copy, pred, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imshow(filename, cv2.resize(img_copy, (200, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
