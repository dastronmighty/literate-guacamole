import os
import shutil


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


def make_folder(path, name):
    p = f"{path}/{name}"
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)

