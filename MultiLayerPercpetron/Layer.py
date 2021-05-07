from MultiLayerPercpetron.Activate import Activate


class Layer:
    """
    A single layer of a Multi-Layer Perceptron
    """

    def __init__(self, act_f):
        """
        :param act_f: The activation function to use
        """
        self.activation = Activate(act_f)

    def linear_forward(self, W, b, A_prev):
        """
        forward propagate through the linear this layer
        :param W: The weights to use
        :param b: the bias('s) to use
        :param A_prev: the previous activation to use
        :return: the
        """
        Z = A_prev @ W + b
        cache = {"W": W, "b": b, "A_prev": A_prev}
        return Z, cache

    def forward(self, W, b, A_prev):
        """
        :param W: Weight Matrix
        :param b: bias
        :param A_prev: Previous input
        :return: Activation, cache
        """
        Z, linear_cache = self.linear_forward(W, b, A_prev)
        A, activation_cache = self.activation.forward(Z)
        return A, {'LINEAR': linear_cache, 'ACTIVATION': activation_cache}

    def linear_backward(self, dZ, cache):
        """
        :param dZ: derivative of the derivative of the error w.r.t the activation
        :param cache: cache
        :return: the derivative of the the weights, biases w.r.t the loss and the derivative of the error w.r.t this layer
        """
        W = cache["W"]
        A_prev = cache["A_prev"]
        n = A_prev.shape[0]
        dW = (1 / n) * A_prev.T @ dZ
        db = (1 / n) * dZ.sum(axis=0, keepdims=True)
        dA_prev = dZ @ W.T
        return dW, db, dA_prev

    def backward(self, dA, cache):
        """
        :param dA: derivative of the error w.r.t the layer before
        :param cache: the cache for this layer
        :return: the derivative of the the weights, biases w.r.t the loss and the derivative of the error w.r.t this layer
        """
        linear_cache = cache['LINEAR']
        activation_cache = cache['ACTIVATION']
        dZ = self.activation.backward(dA, activation_cache)
        dW, db, dA_prev = self.linear_backward(dZ, linear_cache)
        return dW, db, dA_prev
