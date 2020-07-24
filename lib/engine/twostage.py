import os
import torch

import torch.nn as nn
from torch.optim import lr_scheduler

from lib.misc.optimizer import AdamW
from lib.misc.scheduler import WarmUpLR
from lib.misc import util


class TwoStage(object):
    def name(self):
        return 'TwoStage'

    @staticmethod
    def modify_commandline_options(parser, is_test):
        """ modify parser to add command line options, and also change the default values if needed """
        # parser.set_defaults change the "default" values in parser. But command line has higher priority than
        # set_defaults, i.e., set_defaults cannot change the values of command line arguments.
        # Transform parameters (4)
        parser.add_argument('--max_size', type=int, default=1000,
                            help="Max pixel size of the longest side of a scaled input image")
        parser.add_argument('--scales', type=int, default=[600, ], nargs='+',
                            help="The scale is the pixel size of an image's shortest side"
                                 "Scale to use during training (can list multiple scales)")
        parser.add_argument('--pixel_means', type=float, default=[102.9801, 115.9465, 122.7717], nargs='+',
                            help='Pixel mean values (BGR order)')  # todo-note: BGR order
        parser.add_argument('--pixel_stds', type=float, default=[1., 1., 1.], nargs="+",
                            help='size of the last layer feature maps')
        # RPN parameters (9)
        parser.add_argument('--rpn_positive_overlap', default=0.7, type=float,
                            help='IOU >= thresh: positive example. For computing loss')
        parser.add_argument('--rpn_negative_overlap', default=0.3, type=float,
                            help='IOU < thresh: negative example. For computing loss')
        parser.add_argument('--rpn_fg_fraction', default=0.5, type=float,
                            help='Fraction of foreground examples. For computing loss')
        parser.add_argument('--rpn_batch_size', default=256, type=int,
                            help='Total number of the examples used for computing loss')
        parser.add_argument('--rpn_nms_thresh', default=0.7, type=float,
                            help='NMS threshold used on RPN proposals. Removes lower scoring boxes which have an '
                                 'IoU greater than iou_threshold with another (higher scoring) box.')
        parser.add_argument('--rpn_pre_nms_top_n_eval', default=6000, type=int,
                            help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
        parser.add_argument('--rpn_post_nms_top_n_eval', default=300, type=int,
                            help='Number of top scoring boxes to keep after applying NMS to RPN proposals')
        parser.add_argument('--rpn_pre_nms_top_n', default=12000, type=int,
                            help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
        parser.add_argument('--rpn_post_nms_top_n', default=2000, type=int,
                            help='Number of top scoring boxes to keep after applying NMS to RPN proposals')
        # Box parameters (8)
        # inference
        parser.add_argument('--box_score_thresh', default=0.05, type=float,
                            help='Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)')
        parser.add_argument('--box_nms_thresh', default=0.4, type=float,
                            help='Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)')
        parser.add_argument('--box_detections_per_img', default=100, type=int,
                            help='Inference. #boxes per image in test phase.')
        # training
        parser.add_argument('--n_roi_per_img', default=64, type=int,
                            help='Training. Mini-batch size (number of regions of interest [ROIs]')
        parser.add_argument('--fg_fraction', default=0.5, type=float,
                            help='Fraction of mini-batch that is labeled foreground (i.e. class > 0)')
        parser.add_argument('--box_fg_iou_thresh', default=0.5, type=float)
        parser.add_argument('--box_bg_iou_thresh', default=0.5, type=float)
        parser.add_argument('--bbox_reg_weights', default=None)

        return parser

    @staticmethod
    def get_common_args(cfg):
        common_args = dict(
            # Transform parameters (4)
            min_size=cfg.scales, max_size=cfg.max_size,
            image_mean=cfg.pixel_means, image_std=[1., 1., 1.],
            # RPN parameters (9)
            rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n,
            rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_eval,
            rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n,
            rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_eval,
            rpn_nms_thresh=cfg.rpn_nms_thresh,
            rpn_fg_iou_thresh=cfg.rpn_positive_overlap, rpn_bg_iou_thresh=cfg.rpn_negative_overlap,
            rpn_batch_size_per_image=cfg.rpn_batch_size, rpn_positive_fraction=cfg.rpn_fg_fraction,
            # Box parameters (8)
            box_score_thresh=cfg.box_score_thresh, box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
            box_fg_iou_thresh=cfg.box_fg_iou_thresh, box_bg_iou_thresh=cfg.box_bg_iou_thresh,
            box_batch_size_per_image=cfg.n_roi_per_img, box_positive_fraction=cfg.fg_fraction,
            bbox_reg_weights=cfg.bbox_reg_weights,
            # Misc
            eval_gt=cfg.eval_gt, display=cfg.display, cws=cfg.cws
        )
        return common_args

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = cfg.logger
        self.gpu_ids = cfg.gpu_ids
        self.start_epoch = 0
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.device = torch.device(cfg.device)
        self.ckpt_dir = cfg.ckpt_dir
        self.expr_dir = os.path.join(cfg.ckpt_dir, cfg.expr_name)
        self.loss_names = []
        self.acc_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.training = not cfg.is_test
        self.optimizer = {}
        self.lr_scheduler = {}
        self.loss_meter = util.AverageMeter()

    def to_device(self, v):
        try:
            v = v.to(self.device)
        except AttributeError:
            v = v
        return v

    def grouping_params(self, cfg):
        pass

    def set_input(self, input):
        pass

    def forward(self):
        pass

    # load and print networks
    def setup(self, cfg):
        if cfg.load_ckpt:
            self.load_checkpoint(cfg.load_ckpt)
        else:
            print("Not loading checkpoint!")

        self.print_networks(verbose=cfg.model_verbose)

    # make models train mode during training time
    def train(self):
        self.training = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    # make models eval mode during test time
    def eval(self):
        self.training = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self, cfg):
        pass

    # update learning rate (called once every epoch)
    def update_group_lr(self, epoch, n_iter=None):
        """Update Schedulers Dict"""
        for k, scheduler in self.lr_scheduler.items():
            if n_iter is None:
                scheduler.step(epoch)
            else:
                scheduler.step_iter_wise(epoch, n_iter)  # iter-wise warm-up
        # lr = []
        # for k, scheduler in self.lr_scheduler.items():
        #     scheduler.step(epoch)
        #     # suitable for optimizer contains only one params_group
        #     lr.extend(scheduler.get_lr())  # get_lr return a list contain current lr of params_group in scheduler
        #
        # message = 'learning rate = '
        # for item in lr:
        #     message += '%.7f ' % (item)
        #
        # return message

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = dict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_records(self):
        errors_ret = dict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))

        # collect training accuracy together with loss
        for name in self.acc_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                name = 'acc_' + name
                errors_ret[name] = float(getattr(self, name))

        return errors_ret

    @staticmethod
    def decode_losses(losses):
        ret = {
            "acc": losses['acc'],
            "rpn_cls": losses['loss_objectness'],
            "rpn_box": losses['loss_rpn_box_reg'],
            "cls": losses['loss_classifier'],
            "box": losses['loss_box_reg'],
        }
        for k, v in losses.items():
            if k not in ['loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg']:
                ret.update({k.split("loss_")[-1]: v})
        return ret

    def net_to_device(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.to(self.device)

    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    if classname.find('Conv') != -1:
                        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    elif classname.find('Linear') != -1:
                        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                        if m.bias:
                            nn.init.constant_(m.bias.data, 0.0)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    # save models to the disk
    def save_checkpoint(self, epoch):
        save_name = os.path.join(self.expr_dir, 'latest.pth')
        model_state = dict()
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    model_state[name] = net.module.cpu().state_dict()
                else:
                    model_state[name] = net.cpu().state_dict()
                net.to(self.device)

        state = {'epoch': epoch + 1,
                 'state_dict': model_state,
                 'optimizer': {k: o.state_dict() for k, o in self.optimizer.items()},
                 'lr_scheduler': {k: l.state_dict() for k, l in self.lr_scheduler.items()},
                 }
        torch.save(state, save_name)
        self.logger.msg('save model: {}\n'.format(save_name))

    def load_checkpoint(self, load_ckpt):
        '''
        Args:
            load_path: absolute path of the checkpoint

        Returns:
        '''
        self.logger.msg("loading checkpoint %s" % (load_ckpt))
        # checkpoint = torch.load(cfg.load_ckpt, map_location=lambda storage, loc: storage.cuda(cfg.gpu_ids[0]))
        checkpoint = torch.load(self.cfg.load_ckpt, map_location=self.device)  # torch.device('cpu')

        self.start_epoch = checkpoint['epoch']
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = checkpoint['state_dict'][name]
                net.load_state_dict(state_dict, strict=True)

        if not self.cfg.is_test:
            for k, v in checkpoint['optimizer'].items(): self.optimizer[k].load_state_dict(v)
            for k, v in checkpoint['lr_scheduler'].items(): self.lr_scheduler[k].load_state_dict(v)

        print("checkpoint loaded successfully.")
        del checkpoint

    # print network information
    def print_networks(self, verbose):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel() if param.requires_grad else 0
                if verbose:
                    print(net)
                print('#ModelParams : %.2f M' % (num_params / 1e6))

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_scheduler(self, optimizer, lr_policy, lr_decay_milestones, cfg):
        if lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + cfg.epoch_count - cfg.niter) / float(cfg.niter_decay + 1)
                return lr_l

            def lr_lambda(epoch):
                if epoch < lr_decay_milestones[0]:
                    return 1.0
                elif lr_decay_milestones[0] <= epoch < lr_decay_milestones[1]:
                    return 0.33333
                elif lr_decay_milestones[1] <= epoch < lr_decay_milestones[2]:
                    return 0.1
                else:
                    return 0.01

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_milestones, gamma=cfg.lr_decay_gamma)
        elif lr_policy == 'mlt_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=cfg.lr_decay_gamma)
        elif lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)

        if cfg.warmup_epochs > 0:
            return WarmUpLR(
                optimizer, scheduler, cfg.warmup_epochs, cfg.n_batch_per_epoch, cfg.warmup_mode, alpha=cfg.alpha)
        return scheduler

    def get_optimizer(self, optim, params, lr, weight_decay):
        if optim == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim == 'adamw':
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optim == 'amsgrad':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
        elif optim == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim == 'adab':
            # return AdaBound(params, lr=lr, final_lr=0.1, weight_decay=weight_decay)
            raise NotImplementedError
        else:
            raise KeyError("Unsupported optimizer: {}".format(optim))
