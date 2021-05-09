import numpy as np
import os


def add_xor_data(X):
    return X[:, 0] + X[:, 1]


def gen_XOr_data():
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return x, y


def add_sub_sin_helper(x):
    return x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]


def gen_sin_data(seed=42, size=500, train_size=400):
    np.random.seed(seed)
    x = np.random.uniform(-1.0, 1.0, size=(size, 4))
    y = np.sin(add_sub_sin_helper(x))
    y = y.reshape(-1, 1)
    tr_x, tr_y, te_x, te_y = x[0:train_size], y[0:train_size], x[train_size:], y[train_size:]
    return (tr_x, tr_y), (te_x, te_y)


ALPHAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def array_to_letter(t):
    if len(t.shape) == 1:
        idx = [t.argmax()]
    else:
        idx = t.argmax(axis=1)
    return np.array([ALPHAB[i] for i in idx])


def one_hot_letter(letter):
    y = list(np.zeros(26))
    y[ALPHAB.index(letter)] = 1.0
    return y

def process_line(line):
    y = one_hot_letter(line[0])
    x = line[2:]
    x = [float(_) for _ in x.split(",")]
    return x, y

def gen_letter_data(test_percent=0.2):
    full_path = os.path.dirname(os.path.realpath(__file__))
    with open(f"{full_path}/letter-recognition.data") as data_file:
        lines = data_file.readlines()
    x_a, y_a = [], []
    for l in lines:
        x, y = process_line(l)
        x_a.append(x),
        y_a.append(y)
    x, y = np.array(x_a), np.array(y_a)
    train_amt = int(len(x) - int(len(x) * test_percent))
    tr_x, te_x = x[0:train_amt], x[train_amt:]
    tr_y, te_y = y[0:train_amt], y[train_amt:]
    return (tr_x, tr_y), (te_x, te_y)

