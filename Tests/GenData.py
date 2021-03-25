import numpy as np


def gen_XOr_data():
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    Y = np.array([[0],[1],[1],[0]])
    return X, Y


def gen_sin_data(seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1.0, 1.0, size=(500, 4))
    Y = np.sin(X[:,0]-X[:,1]+X[:,2]-X[:,3])
    Y = Y.reshape(-1, 1)
    tr_x, tr_y, te_x, te_y = X[0:400], Y[0:400], X[400:], Y[400:]
    return (tr_x, tr_y), (te_x, te_y)



