import os
from collections import Counter

folder = "Bars"
Y = Counter()

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        txt_path = os.path.join(folder, os.path.splitext(filename)[0] + ".txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "").replace("\r", "")

                if ":" not in content:
                    continue

                words = content.split(":", 1)[1].split()
                Y.update(words)
print(Y)
