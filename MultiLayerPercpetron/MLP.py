from MultiLayerPercpetron.Layer import Layer


class MLP:

    """
    The Multi-Layer-Perceptron Class
    """

    def __init__(self, params):
        """
        :param params: The parameters of the Multi Layer Perceptron class
        """
        self.params = params
        self.layers = []
        for ac in params["activations"]:
            self.layers.append(Layer(ac))
        self.size = len(self.layers)

    def forward(self, X):
        """
        The forward pass through the perceptron
        :param X: The X values to use
        :return: The output and the caches
        """
        caches = []
        A = X
        for i, l in enumerate(self.params["layers"]):
            A, c = self.layers[i].forward(l["W"], l["b"], A)
            caches.append(c)
        return A, caches

    def single_back(self, i, dA, cache):
        """
        The single layer backprop
        :param i: the index of the layer currently going back through
        :param dA: The current error w.r.t the the layer
        :param cache: The layer cache
        :return: the derivatives of the layer
        """
        dW, db, dA_prev = self.layers[i].backward(dA, cache)
        return {
            "dW": dW,
            "db": db,
            "dA_prev": dA_prev
        }

    def backward(self, da_wrt_loss, caches):
        """
        The backwards pass through the MLP
        :param da_wrt_loss: the derivative of the error w.r.t the loss
        :param caches: the chaches from the forward prop
        :return: the gradients of the network
        """
        gradients = {}
        gradients["layers"] = [None for _ in range(self.size)]

        curr_cache = caches[(self.size - 1)]
        final_layer_grads = self.single_back((self.size - 1), da_wrt_loss, curr_cache)
        gradients["layers"][(self.size - 1)] = final_layer_grads

        for i in reversed(range(self.size - 1)):
            curr_cache = caches[i]
            dA_prev = gradients["layers"][i + 1]["dA_prev"]
            layer_i_grads = self.single_back(i, dA_prev, curr_cache)
            gradients["layers"][i] = layer_i_grads

        return gradients

    def predict(self, X, f):
        """
        Predict the output for X
        :param X: the X values as input
        :param f: the function to use to predict
        :return: the predictions
        """
        A, _c = self.forward(X)
        pred = f(A)
        return pred