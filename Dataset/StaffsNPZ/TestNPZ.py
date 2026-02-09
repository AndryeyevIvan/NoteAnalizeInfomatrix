import numpy as np

path = "staffsYOLO(Inverted, 352, 20, 194).npz"

def loadDataset():
    with np.load(path, allow_pickle=True) as f:
        # x_train = (f['images'].astype("float32") / 255.0).tolist()
        # y_train = f['answers'].tolist()

        x_train = f['images'].astype("float32").tolist()
        y_train = f['answers'].astype("float32").tolist()

        print(x_train[0])
        print(y_train[0])

loadDataset()