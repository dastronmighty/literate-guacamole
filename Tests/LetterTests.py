from Data.GenData import gen_letter_data, array_to_letter
from MultiLayerPercpetron.MLP import MLP
from MultiLayerPercpetron.FitModel import fit_model
from MultiLayerPercpetron.Optimisers import ADAM
from MultiLayerPercpetron.Loss import Loss
from MultiLayerPercpetron.paramters import gen_params
from MultiLayerPercpetron.Metrics import multi_class_accuracy, multi_class_f1
from MultiLayerPercpetron.plotter import Plotter
from MultiLayerPercpetron.logger import Logger
from MultiLayerPercpetron.Utils import make_folder


(X_train, Y_train), (X_test, Y_test) = gen_letter_data()

metrics = {
    "acc": multi_class_accuracy,
    "f1": multi_class_f1
}

pred_func = lambda x: array_to_letter(x)

def test_model(num, lr, hidden_layers):
    net = MLP(gen_params(16, hidden_layers, 26, activations=['sigmoid', "softmax"]))
    optim = ADAM(net.params, lr)
    loss_func = Loss("bce")
    logger = Logger("./logs",
                    f"LetterTest{num}.txt",
                    metrics,
                    verbose=10,
                    early_stopping=10)
    plotter = Plotter(f"letterfigs/t{num}/letter test {num}",
                      x_dim_reduction=lambda x: x.sum(axis=1),
                      y_dim_reduction=lambda x: x.argmax(axis=1).flatten())
    net, losses = fit_model(net,
                            X_train,
                            Y_train,
                            32,
                            optim,
                            loss_func,
                            logger,
                            plotter=plotter,
                            epochs=1000,
                            prediction_func=pred_func,
                            y_func=pred_func,
                            x_test=X_test,
                            y_test=Y_test)


make_folder(".", "letterfigs")
test_number = 1
for ls in [[10], [16], [32]]:
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        make_folder("./letterfigs", f"t{test_number}")
        test_model(test_number, lr, ls)
        test_number += 1

