import numpy as np

from GenData import gen_sin_data

from MultiLayerPercpetron.Utils import gen_params, model_summary, get_batches
from MultiLayerPercpetron.MLP import MLP
from MultiLayerPercpetron.FitModel import fit_model
from MultiLayerPercpetron.Optimisers import ADAM, SGD
from MultiLayerPercpetron.Metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

(X_train, Y_train), (X_test, Y_test) = gen_sin_data()
train_batches = get_batches(X_train, Y_train, 32)

hidden_layers = 5
epochs = 10000
size = 32
LR = 0.0001

metrics = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error
}


hidden_layer_sizes = [64, 32, 32, 16, 8, 8, 8, 8]
acts = ['sigmoid']*(hidden_layers) + ["linear"]
params = gen_params(4, hidden_layer_sizes, 1, activations=acts, loss="mse")

net = MLP(params)

print(model_summary(net))

# optim = ADAM(params, LR, 0.8, 0.999)
optim = SGD(LR)

print(f"Hidden Layers = {hidden_layers} - Layer Sizes = {hidden_layer_sizes} - LR = {LR}")
net, losses = fit_model(net,
                        train_batches,
                        optim,
                        epochs,
                        verbose=500,
                        metrics=metrics,
                        X_test=X_test,
                        Y_test=Y_test)

-


