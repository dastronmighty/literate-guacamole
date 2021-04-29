from matplotlib import pyplot as plt


class Plotter:

    def __init__(self,
                 name,
                 x_dim_reduction=(lambda x: x),
                 y_dim_reduction=(lambda x: x),
                 x_label="x",
                 y_label="y",
                 figsize=(10, 10),
                 dpi=100,
                 fontsize=10):
        self.reduce_x = x_dim_reduction
        self.reduce_y = y_dim_reduction
        self.p = "/".join(name.split("/")[0:-1])
        self.name = name.split("/")[-1]
        self.xlab = x_label
        self.ylab = y_label
        self.fs = figsize
        self.dpi = dpi
        self.label_font_size = 30
        self.main_font_size = 28
        self.tick_font_size = 22

    def plot_results(self,
                     model,
                     tag,
                     x_train,
                     y_train,
                     x_test=None,
                     y_test=None):
        train_preds, _ = model.forward(x_train)
        x_train = self.reduce_x(x_train)
        y_train = self.reduce_y(y_train)
        train_preds = self.reduce_y(train_preds)
        self.plot_vs(x_train,
                     y_train,
                     x_train,
                     train_preds,
                     f"{self.name} - {tag} - Training")
        if (x_test is not None) and (y_test is not None):
            test_preds, _ = model.forward(x_test)
            x_test = self.reduce_x(x_test)
            y_test = self.reduce_y(y_test)
            test_preds = self.reduce_y(test_preds)
            self.plot_vs(x_test,
                         y_test,
                         x_test,
                         test_preds,
                         f"{self.name} - {tag} - Testing")

    def plot_vs(self,
                x1,
                y1,
                x2,
                y2,
                plot_name):
        f, axs = plt.subplots(1, 1, figsize=(15, 10))
        axs.set_title(plot_name, fontsize=self.main_font_size)
        axs.set_xlabel(self.xlab, fontsize=self.label_font_size)
        axs.set_ylabel(self.ylab, fontsize=self.label_font_size)
        axs.tick_params(axis='x', length=12, width=4, labelsize=self.tick_font_size)
        axs.tick_params(axis='y', length=12, width=4, labelsize=self.tick_font_size)
        axs.scatter(x1, y1, s=20, c="tab:red")
        axs.scatter(x2, y2, s=20, c="tab:blue")
        axs.legend(["actual", "predicted"], fontsize=self.tick_font_size)
        f.savefig(f"{self.p}/{plot_name}", dpi=self.dpi, bbox_inches='tight', pad_inches=0)

    def plot_mets(self,
             x,
             y1,
             y2,
             type,
             name,
             xlab,
             ylab):
        f, axs = plt.subplots(1, 1, figsize=(15, 10))
        axs.set_title(name, fontsize=self.main_font_size)
        if type == "plot":
            axs.plot(x, y1, c="tab:red", linewidth=5)
            if len(y2) > 1:
                axs.plot(x, y2, c="tab:blue", linewidth=5)
        if type == "scatter":
            axs.scatter(x, y1, c="tab:red", s=20)
            if len(y2) > 1:
                axs.scatter(x, y2, c="tab:blue", s=20)
        axs.set_xlabel(xlab, fontsize=self.label_font_size)
        axs.set_ylabel(ylab, fontsize=self.label_font_size)
        lgd = ["Train"]
        if len(y2) > 0:
            lgd += ["Test"]
        axs.tick_params(axis='x', length=12, width=4, labelsize=self.tick_font_size)
        axs.tick_params(axis='y', length=12, width=4, labelsize=self.tick_font_size)
        axs.legend(lgd, fontsize=self.tick_font_size)
        f.savefig(f"{self.p}/{name}", dpi=self.dpi, bbox_inches='tight', pad_inches=0)
