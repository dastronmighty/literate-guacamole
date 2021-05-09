from Data.GenData import gen_sin_data, add_sub_sin_helper
from MultiLayerPercpetron.MLP import MLP
from MultiLayerPercpetron.FitModel import fit_model
from MultiLayerPercpetron.Optimisers import ADAM
from MultiLayerPercpetron.Loss import Loss
from MultiLayerPercpetron.paramters import gen_params
from MultiLayerPercpetron.Metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from MultiLayerPercpetron.plotter import Plotter
from MultiLayerPercpetron.logger import Logger
from MultiLayerPercpetron.Utils import make_folder


(X_train, Y_train), (X_test, Y_test) = gen_sin_data()

metrics = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error
}


def test_model(model, optim, loss_func, name, logger, epochs=25000):
    plotter = Plotter(name, x_dim_reduction=add_sub_sin_helper)
    net, losses = fit_model(model,
                            X_train,
                            Y_train,
                            32,
                            optim,
                            loss_func,
                            logger,
                            epochs=epochs,
                            plotter=plotter,
                            x_test=X_test,
                            y_test=Y_test)


def test_sin_lr_hidden_layer(name, log_name, lr, hidden_layer_sizes):
    acts = ['sigmoid', "linear"]
    params = gen_params(4, hidden_layer_sizes, 1, activations=acts)
    net = MLP(params)
    optim = ADAM(params, lr)
    loss_func = Loss("mse")
    logger = Logger("./logs",
                    f"{log_name}.txt",
                    metrics,
                    verbose=1000,
                    early_stopping=1024)
    name = f"{name}_{str(lr).replace('.','_')}"
    test_model(net, optim, loss_func, name, logger)


test_number = 1

make_folder(".", "sinfigs")
for layers in [[5], [8], [16]]:
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        make_folder("./sinfigs", f"t{test_number}")
        name = f"sinfigs/t{test_number}/sin test {test_number}"
        log_name = f"SinLog{test_number}"
        test_sin_lr_hidden_layer(name, log_name, lr, layers)
        test_number += 1
