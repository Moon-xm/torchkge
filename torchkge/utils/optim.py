import torch.optim as optim


def optimizer(model, opt_method="sgd", lr=0.1, lr_decay=0, weight_decay=0):

    if opt_method == "Adagrad" or opt_method == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "Adadelta" or opt_method == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_method == "Adam" or opt_method == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return optimizer
