import numpy as np


def rand_init(m, p):
    return np.random.uniform(-1, 1, size=(m, p))


def rand_init_large(m, p):
    return np.random.uniform(-10, 10, size=(m, p))


def he_init(m, p):
    return np.random.normal(0, (np.sqrt(2 / m)), size=(m, p))


def zero_init(m, p):
    return np.zeros((m, p))


def one_init(m, p):
    return np.ones((m, p))


def init_x(m, p, init_func):
    if init_func == "rand":
        return rand_init(m, p)
    elif init_func == "rand_large":
        return rand_init_large(m, p)
    elif init_func == "he":
        return he_init(m, p)
    elif init_func == "one":
        return one_init(m, p)
    elif init_func == "zero":
        return zero_init(m, p)


def layer_activations_gen(params, activations):
    params["activations"] = []
    for i in range(len(params["layers"])):
        act = "linear"
        if activations is not None:
            if len(activations) > i:
                act = activations[i]
        params["activations"].append(act)
    return params


def layer_param_gen(params, layer_sizes, weight_init, bias_init):
    params["layers"] = []
    for i in range(1, len(layer_sizes)):
        params["layers"].append({
            "W": init_x(layer_sizes[i-1], layer_sizes[i], weight_init),
            "b": init_x(1, layer_sizes[i], bias_init)
        })
    return params


def gen_params(in_size,
               hidden_layer_sizes,
               out_size,
               activations=None,
               weight_init="he",
               bias_init="zero", seed=42,):
    np.random.seed(seed)  # for reproducibility
    params = {}
    ls = [in_size] + hidden_layer_sizes + [out_size]
    params = layer_param_gen(params, ls, weight_init, bias_init)
    params = layer_activations_gen(params, activations)
    return params
