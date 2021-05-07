import numpy as np

"""
All the metrics
"""

class confusion_mat:

    def __init__(self, predictions, ground_truth):
        self.tp = (ground_truth == 1)*(predictions == 1)*1
        self.tp = np.sum(self.tp)
        self.tn = (ground_truth == 0)*(predictions == 0)*1
        self.tn = np.sum(self.tn)
        self.fp = (ground_truth == 0)*(predictions == 1)*1
        self.fp = np.sum(self.fp)
        self.fn = (ground_truth == 1)*(predictions == 0)*1
        self.fn = np.sum(self.fn)

    def mat(self):
        return np.array([[self.tp, self.fp],
                         [self.fn, self.tn]])


def accuracy(y_pred, y_true):
    cm = confusion_mat(y_pred, y_true)
    denom = (cm.tp+cm.fn+cm.tn+cm.fp)
    if denom == 0:
        return 0
    else:
        return (cm.tp+cm.tn)/denom


def precision(y_pred, y_true):
    cm = confusion_mat(y_pred, y_true)
    denom = (cm.tp+cm.fp)
    if denom == 0:
        return 0
    else:
        return (cm.tp)/denom


def recall(y_pred, y_true):
    cm = confusion_mat(y_pred, y_true)
    denom = (cm.tp+cm.fn)
    if denom == 0:
        return 0
    else:
        return (cm.tp)/denom


def specificity(y_pred, y_true):
    cm = confusion_mat(y_pred, y_true)
    denom = (cm.tn+cm.fp)
    if denom == 0:
        return 0
    else:
        return (cm.tn)/denom


def fbeta_score(y_pred, y_true, beta):
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    numer = (1+np.power(beta, 2))* prec * rec
    denom = np.power(beta, 2) * prec + rec
    if denom == 0:
        return 0
    else:
        return numer/denom


def f1_score(y_pred, y_true):
    return fbeta_score(y_pred, y_true, 1)


def mean_absolute_error(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_pred, y_true):
    return np.mean(np.power((y_true - y_pred), 2))


def root_mean_squared_error(y_pred, y_true):
    return np.sqrt(np.mean(np.power((y_true - y_pred), 2)))


def r_squared(y_pred, y_true):
    sse = np.nansum(np.power((y_pred, y_true), 2))
    y_true_hat = y_true.mean()
    dev = np.nansum(np.power((y_true-y_true_hat), 2))
    return 1 - (sse/dev)


def multi_class_conf_mat(y_pred, y_true):
    stats = {}

    if len(y_pred) != len(y_true):
        raise RuntimeError(f"length of predictions does not match ground truths")

    labels = np.array(list(set(np.append(y_true, y_pred))))
    for l in labels:
        stats[str(l)] = {
            "tp": 0,
            "fn": 0,
            "fp": 0
        }
    for j, i in zip(y_true, y_pred):
        if j == i:
            stats[str(j)]["tp"] += 1
        else:
            stats[str(j)]["fn"] += 1
            stats[str(i)]["fp"] += 1
    precision_ = 0
    recall_ = 0
    for l in labels:
        pred = stats[str(l)]["tp"]
        total_p = stats[str(l)]["tp"] + stats[str(l)]["fp"]
        total_r = stats[str(l)]["tp"] + stats[str(l)]["fn"]
        precision_ += (pred/total_p) if total_p != 0 else 1
        recall_ += (pred/total_r) if total_r != 0 else 1
    precision_ /= len(labels)
    recall_ /= len(labels)
    return precision_, recall_


def multi_fbeta_score(y_pred, y_true, beta):
    prec, rec = multi_class_conf_mat(y_pred, y_true)
    numer = (1+np.power(beta, 2))* prec * rec
    denom = np.power(beta, 2) * prec + rec
    if denom == 0:
        return 0
    else:
        return numer/denom


def multi_class_accuracy(y_pred, y_true):
    acc = 0
    for j, i in zip(y_true, y_pred):
        if j == i:
            acc += 1
    return acc / len(y_pred)


def multi_class_precision(y_pred, y_true):
    prec, rec = multi_class_conf_mat(y_pred, y_true)
    return prec


def multi_class_recall(y_pred, y_true):
    prec, rec = multi_class_conf_mat(y_pred, y_true)
    return rec


def multi_class_f1(y_pred, y_true):
    return multi_fbeta_score(y_pred, y_true, 1)

