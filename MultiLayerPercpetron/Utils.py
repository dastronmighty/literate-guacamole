import numpy as np

def rand_init(m, p):
    return np.random.uniform(-1, 1, size=(m, p))

def zero_init(m, p):
    return np.zeros((m, p))

def one_init(m, p):
    return np.ones((m, p))

def layer_activations_gen(params, activations):
    params["activations"] = []
    for i in range(len(params["layers"])):
        act = "sigmoid"
        if activations is not None:
            if len(activations) > i:
                act = activations[i]
        params["activations"].append(act)
    return params

def layer_param_gen(params, layer_sizes):
    params["layers"] = []
    for i in range(1, len(layer_sizes)):
        params["layers"].append({
            "W": rand_init(layer_sizes[i-1],layer_sizes[i]),
            "b": zero_init(1,layer_sizes[i])
        })
    return params

def gen_params(in_size, hidden_layer_sizes, out_size, activations=None, loss='mse', seed=None):
    if seed is not None:
        np.random.seed(seed) # for reproducibility
    params = {}
    params["loss"] = loss
    ls = [in_size] + hidden_layer_sizes + [out_size]
    params = layer_param_gen(params, ls)
    params = layer_activations_gen(params, activations)
    return params

def get_batches(X, Y, batch_size=32, seed=42):
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

def model_summary(net):
    params = net.params
    dims = None
    summary = ""
    trainable_params = 0
    for l in range(len(params["layers"])):
        summary += f"layer {l} - {params['activations'][l]}\n"
        dims = params["layers"][l]["W"].shape
        trainable_params += dims[0] * dims[1]
        summary += f"\tWeights = {dims[0]}x{dims[1]}\n"
        trainable_params += dims[1]
        summary += f"\tBiases = 1x{dims[1]}\n"
    in_size = params["layers"][0]["W"].shape[0]
    sum_start = f"Input size : {in_size}\n" + f"Output size : {dims[1]}\n"
    summary = ("="*20)+"\n"+sum_start+("="*20)+"\n"+summary+("="*20)+"\n"
    summary += f"Trainable Parameters : {trainable_params}\n"
    summary += ("="*20)
    return summary


def show_gradients(grads):
    grad_sum = ""
    for i, l in enumerate(grads["layers"]):
        avgdW = l["dW"].mean()
        avgdb = l["db"].mean()
        grad_sum += f"Average dW{i} : {avgdW} - Average db{i} : {avgdb}\n"
    return grad_sum

