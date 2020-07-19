import os
import time
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, expr_dir, display):
        self.display = display
        self.saved = False
        if self.display:
            self.writer = SummaryWriter(expr_dir)

        self.log_name = os.path.join(expr_dir, 'log.txt')
        if not os.path.exists(self.log_name):  # TODO(check)
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def display_current_results(self, prefix2image, iters):
        for prefix, image in prefix2image.items():
            self.writer.add_image('Image/{}'.format(prefix), image, iters)

    # losses: dictionary of error labels and values
    def plot_current_losses(self, losses, iters):
        '''
        Args:
            losses: a dict, {"loss_a": value, "loss_b": value}
            iters: current step
        '''
        self.writer.add_scalars("loss", losses, iters)

    # losses: same format as |losses| of plot_current_losses
    def _print_current_losses(self, epoch, i, nbatch, losses, t):
        message = 'epoch: {:03d} ({:03d}/{}), time: {:.3f} '.format(epoch, i, nbatch, t)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_losses(self, suffix, losses):
        """ suffix and losses can be None """
        message = suffix if suffix is not None else ""

        if losses is not None:
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # message: information to be written in log file
    def msg(self, message):
        assert isinstance(message, str)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # message: information to be written in log file
    def messages(self, message):
        assert isinstance(message, list)
        for msg in message:
            assert isinstance(msg, str)
            print(msg)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % msg)
