import numpy as np

path = r"D:\PythonProjects\NoteAnalize3\Dataset\BarsNPZ\AccidentalsNPZ\barsaccidentals(Inverted, 12, 60, r0).npz"

def load_dataset():
    with np.load(path, allow_pickle=True) as f:
        x_train = f['images'].astype("float32").tolist()
        y_train = f['answers'].astype("float32").tolist()

        return x_train, y_train

