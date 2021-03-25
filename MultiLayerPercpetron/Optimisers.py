import numpy as np


class SGD:

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update_params(self, model, grads):
        for i, gl in enumerate(grads["layers"]):
            dW = gl["dW"]
            db = gl["db"]
            model.params["layers"][i]["W"] -= dW * self.lr
            model.params["layers"][i]["b"] -= db * self.lr
        return model


class SDGMomentum():

    def __init__(self, params, learning_rate, beta):
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
        for i, gl in enumerate(grads["layers"]):
            self.v[i]["dvW"] = (self.beta*self.v[i]["dvW"])+((1-self.beta)*gl["dW"])
            self.v[i]["dvb"] = (self.beta * self.v[i]["dvb"]) + ((1 - self.beta) * gl["db"])
            model.params["layers"][i]["W"] -= (self.v[i]["dvW"] * self.lr)
            model.params["layers"][i]["b"] -= (self.v[i]["dvb"] * self.lr)
        return model
    
    
class ADAM:
    def __init__(self, params, learning_rate, beta1, beta2):
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