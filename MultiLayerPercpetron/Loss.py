import numpy as np


class Loss:

    def __init__(self, loss_func):
        self.loss_func = loss_func

    def sum_absolute_error(self, y, y_pred):
        sae = np.nansum(np.abs((y_pred - y)), axis=0) # sum | ypred - y |
        sae = np.squeeze(sae)
        return sae

    def grad_sum_absolute_error(selfself, y, A_out):
        dl_wrt_Aout = ((A_out - y)>0)*1+((A_out - y)<=0)*-1
        dl_wrt_Aout = np.nansum(dl_wrt_Aout, axis=0)
        return dl_wrt_Aout

    def sum_squared_error(self, y, y_pred):
        sse = np.nansum(((y_pred - y) ** 2), axis=0)
        sse = np.squeeze(sse)
        return sse

    def grad_sum_squared_error(self, y, A_out):
        dl_wrt_Aout = 2*np.nansum(A_out - y, axis=0)
        return dl_wrt_Aout

    def mean_squared_error(self, y, y_pred):
        n = y.shape[0]
        mse = np.nansum(((y_pred - y) ** 2)) / n
        mse = np.squeeze(mse)
        return mse

    def grad_mean_squared_error(self, y, A_out):
        n = y.shape[0]
        dl_wrt_Aout = 2 / n * (np.nansum(A_out - y, axis=0))
        return dl_wrt_Aout

    def binary_cross_entropy(self, y, y_pred):
        eps = np.finfo(float).eps
        n = y.shape[0]
        bce = -(y.T @ np.log(y_pred + eps) + (1 - y).T @ np.log(1 - y_pred + eps)) / n
        bce = np.squeeze(bce)
        return bce

    def grad_binary_cross_entropy(self, y, A_out):
        eps = np.finfo(float).eps
        y_inv = 1 - y
        A_out_inv = 1 - A_out + eps
        dl_wrt_Aout = -((y / A_out) - (y_inv / A_out_inv))
        return dl_wrt_Aout

    def forward(self, y, y_pred):
        if self.loss_func == "sae":
            return self.sum_absolute_error(y, y_pred)
        elif self.loss_func == "sse":
            return self.sum_squared_error(y, y_pred)
        elif self.loss_func == "mse":
            return self.mean_squared_error(y, y_pred)
        elif self.loss_func == "bce":
            return self.binary_cross_entropy(y, y_pred)
        return float('nan')

    def backward(self, y, A_out):
        if self.loss_func == "sae":
            return self.grad_sum_absolute_error(y, A_out)
        elif self.loss_func == "sse":
            return self.grad_sum_squared_error(y, A_out)
        elif self.loss_func == "mse":
            return self.grad_mean_squared_error(y, A_out)
        elif self.loss_func == "bce":
            return self.grad_binary_cross_entropy(y, A_out)
        return float('nan')