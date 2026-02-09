from LedetsNet3 import nn
from DataLoader import load_dataset
import cv2
import numpy as np

x_train, y_train = load_dataset()
net = nn()
filterNet = nn()
net.load(path=r"C:\Users\Иван\Downloads\notes_mlp2_epoch_39_loss_0.03.json")
filterNet.load(path=r"C:\Users\Иван\Downloads\notesFilter_mlp_epoch_39_loss_0.82.json")

x = []
y = []

notes = ["-", "la0", "si0", "do1", "re1", "mi1", "fa1", "sol1", "la1", "si1", "do2", "re2", "mi2"]

xt = []
yt = []

for img_idx in range(len(x_train)):
    window = []

    for i in x_train[img_idx][0]:
        for j in i:
            window.append(j)

    net.struct[0] = window
    net.ForwardPropagation()
    pred = net.struct[-1]

    text = ""

    if max(pred) > 0.0:
        print(notes[pred.index(max(pred))])
        text = notes[pred.index(max(pred))]
    else:
        print("-")
        text = "-"

    img = np.array(x_train[img_idx][0])

    img_uint8 = (img * 255).astype("uint8")

    img_big = cv2.resize(img_uint8, (80, 480), interpolation=cv2.INTER_NEAREST)

    img_big = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_big, text, (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow(f"img", img_big)
    cv2.waitKey(100)

    # for i in pred:
    #     xt.append(i)
    #
    # for i in y_train[img_idx]:
    #     yt.append(i)
    #
    # if len(yt) >= 130:
    #     x.append(xt)
    #     y.append(yt)
    #     xt = []
    #     yt = []

# struct = [130, 50, 50, 130]

# print(x[0])
# print(len(x[0]))
# print(y[0])
# print(len(y[0]))
#
# filterNet = nn(struct, correctAnswers=y, learningRate=0.1, inputLayers=x)
# filterNet.WeightsInit()
# filterNet.BiasesInit()
# filterNet.ErrorsInit()
# filterNet.StartLearning(40, path=r"C:\Users\Иван\Downloads", name="notesFilter_mlp")

