import numpy as np


class Activate:
    """
    Activation Functions
    Includes: linear, sigmoid, tanh, relu, and softmax
    """

    def __init__(self, act_f):
        """
        :param act_f: choose which actiavtion function
        """
        self.act_f = act_f

    def linear_forward(self, Z):
        A = Z
        cache = {"Z": Z}
        return A, cache

    def linear_back(self, dA, cache):
        Z = cache["Z"]
        S, _ = self.linear_forward(Z)
        dZ = dA * np.ones_like(S)
        return dZ

    def sigmoid_forward(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = {"Z": Z}
        return A, cache

    def sigmoid_back(self, dA, cache):
        Z = cache["Z"]
        S, _ = self.sigmoid_forward(Z)
        dZ = dA * S * (1 - S)
        return dZ

    def relu_forward(self, Z):
        A = (Z > 0) * Z
        cache = {"Z": Z}
        return A, cache

    def relu_back(self, dA, cache):
        Z = cache["Z"]
        dZ = dA * (Z >= 0) * 1
        return dZ

    def tanh_forward(self, Z):
        A = np.tanh(Z)
        cache = {"Z": Z}
        return A, cache

    def tanh_back(self, dA, cache):
        Z = cache["Z"]
        t, _ = self.tanh_forward(Z)
        dZ = dA * (1 - (t * t))
        return dZ

    def softmax_forward(self,Z):
        s = len(Z.shape) - 1
        Z_ = Z - np.max(Z)
        A = (np.exp(Z_).T / np.sum(np.exp(Z_), axis=s)).T # only difference
        cache = {"Z": Z}
        return A, cache

    def softmax_back(self, dA, cache):
        # softmax back function for mini-batches
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        Z = cache["Z"]
        ax = len(Z.shape) - 1
        s, _ = self.softmax_forward(Z)
        dZ = s * (dA - (dA * s).sum(axis=ax)[:,None])
        return dZ

    def forward(self, Z):
        """
        Forward prop through the activation
        :param Z: the output of the layer to activate
        :return: the activation of the input
        """
        if self.act_f == 'linear':
            return self.linear_forward(Z)
        elif self.act_f == 'sigmoid':
            return self.sigmoid_forward(Z)
        elif self.act_f == 'relu':
            return self.relu_forward(Z)
        elif self.act_f == 'tanh':
            return self.tanh_forward(Z)
        elif self.act_f == 'softmax':
            return self.softmax_forward(Z)

    def backward(self, dA, cache):
        """
        Backward prop through the activation
        :param dA: the derivate with respect output of the layer after
        :param cache: the cache of the backprop
        :return:  the derivative of the output with respect to the activation
        """
        if self.act_f == 'linear':
            return self.linear_back(dA, cache)
        elif self.act_f == 'sigmoid':
            return self.sigmoid_back(dA, cache)
        elif self.act_f == 'relu':
            return self.relu_back(dA, cache)
        elif self.act_f == 'tanh':
            return self.tanh_back(dA, cache)
        elif self.act_f == 'softmax':
            return self.softmax_back(dA, cache)