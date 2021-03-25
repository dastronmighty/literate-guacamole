import numpy as np

from GenData import gen_XOr_data

from MultiLayerPercpetron.Utils import gen_params, get_batches, model_summary
from MultiLayerPercpetron.MLP import MLP
from MultiLayerPercpetron.FitModel import fit_model
from MultiLayerPercpetron.Optimisers import SGD, SDGMomentum, ADAM
from MultiLayerPercpetron.Metrics import accuracy, precision, recall


def test_XOr_out(out):
    if np.equal(out, np.array([[0],[1],[1],[0]])).all():
        print(f"Output:\n{out}")
        print("XOr Test Passed!")
    else:
        print(f"Output:\n{out}")
        print("XOr Test Failed!")

def train_XOr(net):
    X, Y = gen_XOr_data()
    print("Training...")
    batches = get_batches(X, Y, 2)
    optim = SGD(0.1)
    metrics = {
        "Acc.":accuracy,
        "Precision": precision,
        "Recall": recall
    }
    pfunc = lambda x: (x>0.5)*1
    net, losses = fit_model(net, batches, optim, 10000, metrics=metrics, verbose=1000, pfunc=pfunc)
    print("Done!")
    test_XOr_out(net.predict(X, pfunc))


def Test_XOr_1():
    print("Test 1 - small model")
    params = gen_params(2, [8], 1, loss="bce")
    net = MLP(params)
    print(model_summary(net))
    train_XOr(net)


def Test_XOr_2():
    print("Test 2 - bigger model")
    acts = ["tanh", "sigmoid", "tanh", "sigmoid"]
    params = gen_params(2, [8, 16, 8], 1, activations=acts, loss="bce")
    net = MLP(params)
    print(model_summary(net))
    train_XOr(net)


Test_XOr_1()
Test_XOr_2()
