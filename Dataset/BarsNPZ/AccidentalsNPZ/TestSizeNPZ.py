import os
from collections import Counter

folder = "Accidentals"
Y = Counter()

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        txt_path = os.path.join(folder, os.path.splitext(filename)[0] + ".txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "").replace("\r", "")

                if ":" not in content:
                    continue

                dWords = content.split(":", 1)[1]
                if len(dWords) > 0:
                    print(dWords)
                words = []
                for i in range(len(dWords)):
                    if dWords[i] == "б" and i != len(dWords) - 1:
                        if dWords[i + 1] == "к":
                            words.append("бк")
                            continue
                    elif dWords[i] == "к":
                        continue
                    elif dWords[i] != "":
                        words.append(dWords[i])
                if len(words) > 0:
                    print(words)
                Y.update(words)
print(Y)
