from MultiLayerPercpetron.Utils import model_summary

import numpy as np


class Logger:

    """
    The Logger class used to log the output
    """

    def __init__(self,
                 path,
                 name,
                 metrics,
                 verbose=None,
                 verbose_precision=6,
                 early_stopping=None):
        """
        :param path: Path
        :param name: The name of the experiment
        :param metrics: the metrics to use
        :param verbose: Whether or not to also print to the console
        :param verbose_precision: the precision to print the floats to
        :param early_stopping: whether or not to use early stopping
        """
        self.fp = f"{path}/{name}"
        with open(self.fp, "w") as f:
            f.write("")
        self.metrics = metrics
        self.losses = []
        self.test_losses = []
        self.train_metrics = {}
        self.test_metrics = {}
        for k in metrics.keys():
            self.train_metrics[k] = []
            self.test_metrics[k] = []
        self.early_stopping = early_stopping
        self.curr_epoch = 0
        self.verbose = verbose
        self.verbose_precision = verbose_precision

    def log_to_file(self, s):
        "Log a string to a file"
        with open(self.fp, "a") as f:
            f.write(f"{s}\n")

    def update_epoch(self, epoch, test_available):
        """
        Update an epoch
        :param epoch: current epoch
        :param test_available: is test data available
        """
        self.curr_epoch = epoch
        for k in self.metrics.keys():
            self.train_metrics[k].append(0)
            if test_available:
                self.test_metrics[k].append(0)
        self.losses.append(0)
        if test_available:
            self.test_losses.append(0)

    def log_summary(self, model):
        """
        :param model: the model
        """
        summ = model_summary(model)
        self.log_to_file(summ)
        print(summ)

    def log_loss(self, loss, train=True):
        """
        Log a loss
        :param loss: the loss to log
        :param train: is the loss a training loss or not
        """
        if train:
            self.losses[self.curr_epoch] += loss
        else:
            self.test_losses[self.curr_epoch] += loss

    def log_mets(self, x, y, train=True):
        """
        get the metrics from a given x, y
        :param x: the x values to use
        :param y: the y values to use
        :param train: whether the training data is for training ot testing
        """
        for k in self.metrics.keys():
            if train:
                self.train_metrics[k][self.curr_epoch] += self.metrics[k](x, y)
            else:
                self.test_metrics[k][self.curr_epoch] += self.metrics[k](x, y)

    def compress_stats(self, num_batches, train=True):
        """
        compress the stats for all the batches into the average stats for the epoch
        :param num_batches: the number of batches we are using
        :param train: whether it is training or not
        """
        for k in self.metrics.keys():
            if train:
                self.train_metrics[k][self.curr_epoch] /= num_batches
            else:
                self.test_metrics[k][self.curr_epoch] /= num_batches
        if train:
            self.losses[self.curr_epoch] /= num_batches
        else:
            self.test_losses[self.curr_epoch] /= num_batches

    def check_early_stopping(self):
        """
        Check if its time to stop early
        :return: return if we should stop or not
        """
        stop = False
        if self.early_stopping is not None:
            if len(self.test_losses) > self.early_stopping + 1:
                bigger_than, stop = 0, False
                for i in range(2, 2 + self.early_stopping):
                    bigger_than += 1 if self.test_losses[-1] > self.test_losses[-i] else 0
                if bigger_than == self.early_stopping:
                    stop = True
                    s = f"Early stopping on epoch {self.curr_epoch}"
                    self.log_to_file(s)
                    print(s)
        return stop

    def verbose_epoch(self, epoch):
        """
        whether to print on this epoch
        :param epoch: The current epoch
        """
        return epoch % self.verbose == 0

    def log(self, epoch, epochs, test=False):
        """
        Log the epoch stats
        :param epoch: the current epoch
        :param epochs: the total number of epochs
        :param test: whether the data is train or not
        """
        if self.verbose_epoch(epoch):
            ve = ""
            last_loss = self.losses[-1]
            ve = f"Epoch {str(epoch).rjust(self.verbose_precision)}/{str(epochs).ljust(self.verbose_precision)}"
            ve += f" - Last Loss : {np.round(last_loss, self.verbose_precision)}"
            avg_loss = np.array(self.losses).mean()
            ve += f" - Avg. Loss : {np.round(avg_loss, self.verbose_precision)}"
            if test:
                last_test_loss = self.test_losses[-1]
                ve += f" - Last Test Loss : {np.round(last_test_loss, self.verbose_precision)}"
                avg_test_loss = np.array(self.test_losses).mean()
                ve += f" - Avg. Test Loss : {np.round(avg_test_loss, self.verbose_precision)}"
            for k in self.metrics.keys():
                lm = self.train_metrics[k][-1]
                mm = np.array(self.train_metrics[k]).mean()
                ve += f" - Last {k} : {lm} - Avg. {k} : {mm}"
                if test:
                    lm = self.test_metrics[k][-1]
                    mm = np.array(self.test_metrics[k]).mean()
                    ve += f" - Last test {k} : {lm} - Avg. test {k} : {mm}"
            self.log_to_file(ve)
            if self.verbose is not None:
                print(ve)

    def get_loss_to_plot(self):
        """
        get the losses to plot
        :return: the losses to plot along with the plot config
        """
        x = [x for x in range(len(self.losses))]
        y1 = self.losses
        y2 = self.test_losses
        t = "plot"
        name = f"Train Loss"
        if len(y2) > 0:
            name += " vs Test Loss"
        xlab = "Epochs"
        ylab = "Loss"
        return x, y1, y2, t, name, xlab, ylab

    def get_met_to_plot(self, met):
        """
        Get the metircs to plot
        :param met: the metric to get the logs for
        :return: The mats to plot along with the plot config
        """
        x = [x for x in range(len(self.train_metrics[met]))]
        y1 = self.train_metrics[met]
        y2 = self.test_metrics[met]
        t = "plot"
        name = f"Train {met}"
        if len(y2) > 0:
            name += f" vs Test {met}"
        xlab = "Epochs"
        ylab = "Loss"
        return x, y1, y2, t, name, xlab, ylab

