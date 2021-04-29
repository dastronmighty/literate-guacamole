from MultiLayerPercpetron.Optimisers import get_batches


def fit_model(model,
              x_train,
              y_train,
              batch_size,
              optim,
              loss_func,
              logger,
              epochs=1000,
              prediction_func=(lambda x: x),
              y_func=(lambda x: x),
              plotter=None,
              x_test=None,
              y_test=None):

    logger.log_summary(model)
    batches = get_batches(x_train, y_train, batch_size)
    test_available = (x_test is not None) and (y_test is not None)

    test_batches = None
    if test_available:
        test_batches = get_batches(x_test, y_test, batch_size)

    if plotter is not None:
        plotter.plot_results(model, f"Before Trained", x_train, y_train, x_test, y_test)

    for epoch in range(epochs + 1):
        logger.update_epoch(epoch, test_available)
        for xb, yb, in batches:
            a_out, caches = model.forward(xb)
            i_loss = loss_func.forward(yb, a_out)
            logger.log_loss(i_loss)
            logger.log_mets(prediction_func(a_out), y_func(yb))
            optim.backwards_step(yb, a_out, model, loss_func, caches)
        logger.compress_stats(len(batches))

        if test_available:
            for xb, yb, in test_batches:
                test_a_out, _ = model.forward(xb)
                i_loss = loss_func.forward(yb, test_a_out)
                logger.log_loss(i_loss, False)
                logger.log_mets(prediction_func(test_a_out), y_func(yb), False)
            logger.compress_stats(len(test_batches), False)

        logger.log(epoch, epochs, test_available)

        if plotter is not None:
            if logger.verbose_epoch(epoch):
                plotter.plot_results(model,
                                     f"Training epoch {epoch} Loss {str(logger.losses[-1]).replace('.', '_')[0:4]}",
                                     x_train,
                                     y_train,
                                     x_test,
                                     y_test)

        if logger.check_early_stopping():
            break

    if plotter is not None:
        plotter.plot_results(model, f"Trained", x_train, y_train, x_test, y_test)
        plotter.plot_mets(*logger.get_loss_to_plot())
        for k in logger.metrics.keys():
            plotter.plot_mets(*logger.get_met_to_plot(k))

    return model, logger.losses


