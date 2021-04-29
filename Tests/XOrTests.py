from Data.GenData import gen_XOr_data, add_xor_data
from MultiLayerPercpetron.paramters import gen_params
from MultiLayerPercpetron.MLP import MLP
from MultiLayerPercpetron.FitModel import fit_model
from MultiLayerPercpetron.Loss import Loss
from MultiLayerPercpetron.Optimisers import SGD
from MultiLayerPercpetron.Metrics import accuracy, precision, recall
from MultiLayerPercpetron.plotter import Plotter
from MultiLayerPercpetron.logger import Logger
from MultiLayerPercpetron.Utils import make_folder

import numpy as np


metrics = {
    "Acc.": accuracy,
    "Precision": precision,
    "Recall": recall
}


def test_XOr_out(out):
    if np.equal(out, np.array([[0],[1],[1],[0]])).all():
        print(f"Output:\n{out}")
        print("XOr Test Passed!")
    else:
        print(f"Output:\n{out}")
        print("XOr Test Failed!")


def train_XOr(net, logger, loss_func, name):
    X, Y = gen_XOr_data()
    print("Training...")
    optim = SGD(net, 0.1)
    plotter = Plotter(name, add_xor_data)
    pfunc = lambda x: (x > 0.5)*1
    net, losses = fit_model(net,
                            X,
                            Y,
                            2,
                            optim,
                            loss_func,
                            logger,
                            epochs=10000,
                            plotter=plotter,
                            prediction_func=pfunc)
    print("Done!")
    test_XOr_out(net.predict(X, pfunc))


def Test_XOr_1():
    print("Test 1 - small model")
    acts = ["sigmoid", "sigmoid"]
    params = gen_params(2, [3], 1, activations=acts)
    net = MLP(params)
    loss_func = Loss("bce")
    logger = Logger("./logs",
                    "xorLog1.txt",
                    metrics,
                    verbose=1000)
    train_XOr(net, logger, loss_func, "xorfigs/t1/xor test 1")


def Test_XOr_2():
    print("Test 2 - bigger model")
    acts = ["tanh", "sigmoid", "tanh", "sigmoid"]
    params = gen_params(2, [6], 1, activations=acts)
    net = MLP(params)
    loss_func = Loss("bce")
    logger = Logger("./logs",
                    "xorLog2.txt",
                    metrics,
                    verbose=1000)
    train_XOr(net, logger, loss_func, "xorfigs/t2/xor test 2")


make_folder(".", "xorfigs")
make_folder("./xorfigs", "t1")
Test_XOr_1()
make_folder("./xorfigs", "t2")
Test_XOr_2()
