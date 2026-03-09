import numpy as np
import torch


class LrScheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5 * \
            (base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))

        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group["name"] == "predictor":
                param_group["lr"] = self.base_lr
            else:
                lr = param_group["lr"] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ("module.predictor", "predictor")
    parameters = [{
        "name": "base",
        "params": [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        "lr": lr
    }, {
        "name": "predictor",
        "params": [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        "lr": lr
    }]

    if name == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def init_lrscheduler(model, total_epochs, dataloader):
    warmup_epochs = 10
    warmup_lr = 0
    base_lr = 0.03
    final_lr = 0
    momentum = 0.9
    weight_decay = 0.0005
    batch_size = 64

    optimizer = get_optimizer(
        "sgd", model,
        lr=base_lr*batch_size/256,
        momentum=momentum,
        weight_decay=weight_decay)

    lr_scheduler = LrScheduler(
        optimizer, warmup_epochs, warmup_lr*batch_size/256,
        total_epochs, base_lr*batch_size/256, final_lr*batch_size/256,
        len(dataloader),
        constant_predictor_lr=True
    )
    return optimizer, lr_scheduler
