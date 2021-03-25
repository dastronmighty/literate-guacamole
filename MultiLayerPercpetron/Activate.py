import numpy as np


class Activate:

    def __init__(self, act_f):
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

    def softmax_forward(self, Z):
        e = np.exp(Z)
        A = e / np.sum(Z, axis=1)
        cache = {"Z": Z}
        return A, cache

    def softmax_back(self, dA, cache):
        Z = cache["Z"]
        sm, _ = self.softmax_forward(Z)
        s = sm.reshape(-1, 1)
        dZ = dA * (np.diagflat(s) - np.dot(s, s.T))
        return dZ

    def forward(self, Z):
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

