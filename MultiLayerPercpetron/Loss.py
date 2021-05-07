import numpy as np

epsilon = np.finfo(float).eps


class Loss:

    def __init__(self, loss_func):
        """
        :param loss_func: The loss funciton to use
        """
        self.loss_func = loss_func

    def sum_absolute_error(self, y, y_pred):
        """
        the ablsolute error
        """
        sae = np.nansum(np.abs((y_pred - y)), axis=0)
        sae = np.squeeze(sae)
        return sae

    def grad_sum_absolute_error(self, y, a_out):
        dl_wrt_Aout = ((a_out - y) > 0)*1 + ((a_out - y) <= 0)*-1
        dl_wrt_Aout = np.nansum(dl_wrt_Aout, axis=0)
        return dl_wrt_Aout

    def sum_squared_error(self, y, y_pred):
        sse = np.nansum(((y_pred - y) ** 2), axis=0)
        sse = np.squeeze(sse)
        return sse

    def grad_sum_squared_error(self, y, a_out):
        dl_wrt_Aout = 2*np.nansum(a_out - y, axis=0)
        return dl_wrt_Aout

    def mean_squared_error(self, y, y_pred):
        n = y.shape[0]
        mse = np.nansum(((y_pred - y) ** 2)) / (n + epsilon)
        mse = np.squeeze(mse)
        return mse

    def grad_mean_squared_error(self, y, a_out):
        n = y.shape[0]
        dl_wrt_Aout = (2 / n+epsilon) * np.nansum(a_out - y, axis=0)
        return dl_wrt_Aout

    def cross_ent_helper(self, y_pred):
        y_pred = np.clip(y_pred, a_min=epsilon, a_max=1-epsilon)
        return y_pred

    def binary_cross_entropy(self, y, y_pred):
        z = self.cross_ent_helper(y_pred)
        m = y.shape[1]  # m -> number of examples in the batch
        bce = (1 / m) * np.sum(-y*np.log(z) - (1-y)*np.log(1-z))
        bce = np.squeeze(bce)
        return bce

    def grad_binary_cross_entropy(self, y, a_out):
        a_out = self.cross_ent_helper(a_out)
        m = y.shape[1]  # m -> number of examples in the batch
        dbce_wrt_Aout = (1/m) * (-(y/a_out+epsilon) + ((1-y)/(1-a_out+epsilon)))
        return dbce_wrt_Aout

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

    def backward(self, y, a_out):
        if self.loss_func == "sae":
            return self.grad_sum_absolute_error(y, a_out)
        elif self.loss_func == "sse":
            return self.grad_sum_squared_error(y, a_out)
        elif self.loss_func == "mse":
            return self.grad_mean_squared_error(y, a_out)
        elif self.loss_func == "bce":
            return self.grad_binary_cross_entropy(y, a_out)
        return float('nan')

