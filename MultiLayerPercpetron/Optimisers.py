import numpy as np


def get_batches(X, Y, batch_size=32, seed=42):
    """
    Get randomly mixed batches from an input data
    :param X: The X values to make into batch
    :param Y: The labels values to make into the batches also
    :param batch_size: the size of the batches to make
    :param seed: the seed to use for random shuffling
    :return: the batcnes
    """
    np.random.seed(seed)
    n = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(n))
    X_perm = X[permutation]
    Y_perm = Y[permutation]

    count = int(np.floor(n / batch_size)) # number of full batches we cna make
    for i in range(count):
        X_mini_batch = X_perm[(i * batch_size):((i + 1) * batch_size)]
        Y_mini_batch = Y_perm[(i * batch_size):((i + 1) * batch_size)]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)

    if n % batch_size != 0:
        X_mini_batch = X_perm[(count * batch_size):]
        Y_mini_batch = Y_perm[(count * batch_size):]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)
    return mini_batches


class Optimiser:

    """
    The optimiser abstract class
    """
    
    def __init__(self, learning_rate, name):
        """
        :param learning_rate: the learning rate
        :param name: the name of the optimiser
        """
        self.lr = learning_rate
        self.name = name

    def update_params(self, model, grads):
        pass

    def backwards_step(self, y, a_out, model, loss, cache):
        da_wrt_loss = loss.backward(y, a_out)
        grads = model.backward(da_wrt_loss, cache)
        self.update_params(model, grads)


class SGD(Optimiser):

    def __init__(self, params, learning_rate):
        super(SGD, self).__init__(learning_rate, "SGD")

    def update_params(self, model, grads):
        """
        Update the parameters of the model
        :param model: the model
        :param grads: the gradients to use
        :return: the updated gradient
        """
        for i, gl in enumerate(grads["layers"]):
            dW = gl["dW"]
            db = gl["db"]
            model.params["layers"][i]["W"] -= dW * self.lr
            model.params["layers"][i]["b"] -= db * self.lr
        return model


class SGDMomentum(Optimiser):

    def __init__(self, params, learning_rate, beta=0.9):
        super(SGDMomentum, self).__init__(learning_rate, "SGDMomentum")
        self.lr = learning_rate
        self.beta = beta
        self.v = []
        for la in params["layers"]:
            dv = {
                "dvW": np.zeros_like(la["W"]),
                "dvb": np.zeros_like(la["b"])
            }
            self.v.append(dv)

    def update_params(self, model, grads):
        """
        Update the parameters of the model
        :param model: the model
        :param grads: the gradients to use
        :return: the updated gradient
        """
        for i, gl in enumerate(grads["layers"]):
            self.v[i]["dvW"] = (self.beta*self.v[i]["dvW"])+((1-self.beta)*gl["dW"])
            self.v[i]["dvb"] = (self.beta * self.v[i]["dvb"]) + ((1 - self.beta) * gl["db"])
            model.params["layers"][i]["W"] -= (self.v[i]["dvW"] * self.lr)
            model.params["layers"][i]["b"] -= (self.v[i]["dvb"] * self.lr)
        return model
    
    
class ADAM(Optimiser):

    def __init__(self, params, learning_rate, beta1=0.9, beta2=0.999):
        super(ADAM, self).__init__(learning_rate, "ADAM")
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = []
        self.s = []
        self.t = 0
        self.eps = 1e-8
        for la in params["layers"]:
            dv = {
                "dvW": np.zeros_like(la["W"]),
                "dvb": np.zeros_like(la["b"])
            }
            ds = {
                "dsW": np.zeros_like(la["W"]),
                "dsb": np.zeros_like(la["b"])
            }
            self.v.append(dv)
            self.s.append(ds)

    def update_params(self, model, grads):
        """
        Update the parameters of the model
        :param model: the model
        :param grads: the gradients to use
        :return: the updated gradient
        """
        self.t += 1

        for i, gl in enumerate(grads["layers"]):
            self.v[i]["dvW"] = (self.beta1 * self.v[i]["dvW"]) + ((1-self.beta1)*gl["dW"])
            self.v[i]["dvb"] = (self.beta1 * self.v[i]["dvb"]) + ((1 - self.beta1) * gl["db"])
            self.s[i]["dsW"] = (self.beta2 * self.s[i]["dsW"]) + ((1 - self.beta2) * np.power(gl["dW"], 2))
            self.s[i]["dsb"] = (self.beta2 * self.s[i]["dsb"]) + ((1 - self.beta2) * np.power(gl["db"], 2))
            vbarW = self.v[i]["dvW"] / (1 - np.power(self.beta1, self.t))
            vbarb = self.v[i]["dvb"] / (1 - np.power(self.beta1, self.t))
            sbarW = self.s[i]["dsW"] / (1 - np.power(self.beta2, self.t))
            sbarb = self.s[i]["dsb"] / (1 - np.power(self.beta2, self.t))
            wGrad = vbarW/np.sqrt((sbarW+self.eps))
            bGrad = vbarb/np.sqrt((sbarb+self.eps))
            model.params["layers"][i]["W"] -= (wGrad * self.lr)
            model.params["layers"][i]["b"] -= (bGrad * self.lr)

        return model

