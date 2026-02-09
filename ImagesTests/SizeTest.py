import os
import cv2

img_path = r"/Dataset/ClassesNPZ\Classes\0_piece_3.jpg"

img = cv2.imread(img_path)
#img = cv2.resize(img, (152, 152), interpolation=cv2.INTER_AREA)
import cv2

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

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = resize_with_padding_white(img, 200, 400)
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = 255 - img
cv2.imwrite("img.png", img)
cv2.imshow("test", img)
cv2.waitKey(0)