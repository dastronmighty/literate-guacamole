import numpy as np

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
