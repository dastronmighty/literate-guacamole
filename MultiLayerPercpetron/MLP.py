from MultiLayerPercpetron.Layer import Layer
from MultiLayerPercpetron.Loss import Loss


class MLP:

    def __init__(self, params):
        self.params = params
        self.loss = Loss(params["loss"])
        self.layers = []
        for ac in params["activations"]:
            self.layers.append(Layer(ac))
        self.size = len(self.layers)

    def forward(self, X, Y=None):
        caches = []
        A = X
        for i, l in enumerate(self.params["layers"]):
            A, c = self.layers[i].forward(l["W"], l["b"], A)
            caches.append(c)
        loss = None
        if Y is not None:
            loss = self.loss.forward(Y, A)
        return A, loss, caches

    def single_back(self, i, dA, cache):
        dW, db, dA_prev = self.layers[i].backward(dA, cache)
        return {
            "dW": dW,
            "db": db,
            "dA_prev": dA_prev
        }

    def backward(self, A_out, Y, caches):
        gradients = {}
        gradients["layers"] = [None for _ in range(self.size)]

        dA_wrt_Loss = self.loss.backward(Y, A_out)
        gradients["dA_wrt_Loss"] = dA_wrt_Loss

        curr_cache = caches[(self.size - 1)]
        final_layer_grads = self.single_back((self.size - 1), dA_wrt_Loss, curr_cache)
        gradients["layers"][self.size - 1] = final_layer_grads

        for i in reversed(range(self.size - 1)):
            curr_cache = caches[i]
            dA_prev = gradients["layers"][i + 1]["dA_prev"]
            layer_i_grads = self.single_back(i, dA_prev, curr_cache)
            gradients["layers"][i] = layer_i_grads

        return gradients

    def predict(self, X, f):
        A, _l, _c = self.forward(X)
        pred = f(A)
        return pred