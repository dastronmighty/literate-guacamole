import numpy as np

from MultiLayerPercpetron.Utils import show_gradients
from MultiLayerPercpetron.Metrics import accuracy

def fit_model(model,
              batches,
              optim,
              epochs=1000,
              metrics={},
              pfunc=(lambda x: x),
              file=None,
              verbose=None,
              verbose_precision=6,
              X_test=None, Y_test=None):
    losses = []
    train_metrics = {}
    test_metrics = {}
    for k in metrics.keys():
        train_metrics[k] = []
        test_metrics[k] = []

    for i in range(epochs + 1):
        for xb, yb, in batches:
            A, loss, caches = model.forward(xb, yb)
            grads = model.backward(A, yb, caches)
            optim.update_params(model, grads)
            losses.append(loss)

            for k in metrics.keys():
                train_metrics[k].append(metrics[k](pfunc(A), yb))

        if X_test is not None and Y_test is not None:
            for k in metrics.keys():
                test_metrics[k].append(metrics[k](model.predict(X_test, pfunc), Y_test))

        if verbose is not None:
            if i % verbose == 0 and i > 1:
                ve = f"Epoch {str(i).rjust(verbose_precision)}/{str(epochs).ljust(verbose_precision)}"
                lastloss = losses[-1]
                ve += f" - Last Loss : {np.round(lastloss, verbose_precision)}"
                loss = np.array(losses).mean()
                ve += f" - Avg. Loss : {np.round(loss, verbose_precision)}"

                for k in metrics.keys():
                    lm = train_metrics[k][-1]
                    mm = np.array(train_metrics[k]).mean()
                    ve += f" - Last {k} : {lm} - Avg. {k} : {mm}"

                if X_test is not None and Y_test is not None:
                    for k in metrics.keys():
                        lm = test_metrics[k][-1]
                        mm = np.array(test_metrics[k]).mean()
                        ve += f" - Last test {k} : {lm} - Avg. test {k} : {mm}"

                if file is not None:
                    file.write(ve+"\n")
                else:
                    print(ve)

    return model, losses
