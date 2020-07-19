from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    def __init__(
            self, optimizer, scheduler, warmup_epochs, total_batches, mode="linear", alpha=0.01, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.n_batch_per_epoch = total_batches

        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.mode == "linear":
                beta = self.last_epoch / float(self.warmup_epochs)  # 0 -> 1
                factor = self.alpha * (1 - beta) + beta  # beta -> 1
            elif self.mode == "constant":
                factor = self.alpha
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in self.base_lrs]  # initial lr
            # return [factor * base_lr for base_lr in cold_lrs]
        else:
            self.scheduler.last_epoch = self.last_epoch - self.warmup_epochs  # NOTE: important
            cold_lrs = self.scheduler.get_lr()

            return cold_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr_iter_wise(self, n_iter):
        if self.last_epoch < self.warmup_epochs:
            if self.mode == "linear":
                beta = n_iter / float(self.warmup_epochs * self.n_batch_per_epoch)  # 0 -> 1
                factor = self.alpha * (1 - beta) + beta  # alpha -> 1
            elif self.mode == "constant":
                factor = self.alpha
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in self.base_lrs]  # initial lr
            # return [factor * base_lr for base_lr in cold_lrs]
        else:
            self.scheduler.last_epoch = self.last_epoch - self.warmup_epochs  # NOTE: important
            cold_lrs = self.scheduler.get_lr()

            return cold_lrs

    def step_iter_wise(self, epoch=None, n_iter=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr_iter_wise(n_iter)):
            param_group['lr'] = lr


def update_learning_rate(scheduler, epoch):
    scheduler.step(epoch)
    # suitable for optimizer contains only one params_group
    lr = scheduler.get_lr()  # get_lr return a list contain current lr of params_group in scheduler
    message = 'learning rate = '
    for item in lr:
        message += '%.7f ' % (item)
    return message


def update_group_lr(schedulers, epoch):
    """Update Schedulers Dict"""
    lr = []
    for k, scheduler in schedulers.items():
        scheduler.step(epoch)
        # suitable for optimizer contains only one params_group
        lr.extend(scheduler.get_lr())  # get_lr return a list contain current lr of params_group in scheduler

    message = 'learning rate = '
    for item in lr:
        message += '%.7f ' % (item)

    return message


def init_scheduler(optimizer, lr_policy, lr_decay_milestones, opt):
    if lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_milestones, gamma=opt.lr_decay_gamma)
    elif lr_policy == 'mlt_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=opt.lr_decay_gamma)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epoch, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler
