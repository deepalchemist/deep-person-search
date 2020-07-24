import os
from math import ceil
from scipy.misc import imsave

import torch
import torchvision.utils as vutils

from lib.detector.OIM import OIM
from lib.engine.twostage import TwoStage
from lib.misc import util
from lib.misc import visual as vis


class oim(TwoStage):
    """
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --benchmark prw --batch_size 5 \
    --backbone oim --cls_type oim \
    --lr 0.003 --reid_lr 0.003 --uni_optim --warmup_epochs 1 --max_epoch 7 \
    --suffix "" \
    """

    def name(self):
        return 'OIM'

    @staticmethod
    def modify_commandline_options(parser, is_test):
        # common arguments
        parser = TwoStage.modify_commandline_options(parser, is_test)
        # specific arguments
        parser.add_argument('--alpha', type=float, default=0.01, help='warm-up lower lr factor')
        parser.add_argument('--cat_c4', action='store_true', help="concat cov4 and cov5 features as re-ID feature")
        parser.add_argument('--cls_type', type=str, default='oim', help='type of classifier')
        parser.add_argument('--reid_lr', type=float, default=3.5e-4, help='lr of re-ID parameters')
        parser.add_argument('--det_thr', type=float, default=0.5)
        parser.add_argument('--uni_optim', action='store_true', help="Differs optimizers for detection and re-ID.")
        parser.add_argument('--use_mask', action='store_true', help="enable segmentation head")

        cfg = parser.parse_args()
        parser.set_defaults(num_train_ids=5532 if cfg.benchmark == "ssm" else 483)

        # initialize experiment name
        model_name = cfg.backbone
        model_name += 'cat' if cfg.cat_c4 else ""
        model_name += "fs" if cfg.use_mask else ""  # foreground segmentation
        model_name += cfg.cls_type
        augment = "b{}".format(cfg.batch_size)
        augment += "uo" if cfg.uni_optim else "do"
        augment += "wu" if cfg.warmup_epochs > 0 else ""
        augment += "cg" if cfg.clip_grad else ""
        augment += "ba" if cfg.bg_aug else ""

        if not is_test:
            parser.set_defaults(expr_name="{}*{}*{}".format(cfg.benchmark, model_name, augment))

        return parser

    def __init__(self, cfg):
        TwoStage.__init__(self, cfg)

        # specify the training losses you want to print out.
        # The program will call base_model.get_current_losses
        # if cfg.display:
        #     self.visual_names.append('vis_inp')

        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['detector']

        # detector
        common_kwargs = self.get_common_args(cfg)
        self.detector = OIM(
            num_classes=2,
            # re-ID
            num_train_pids=cfg.num_train_ids, cls_type=cfg.cls_type, cat_c4=cfg.cat_c4,
            **common_kwargs
        )
        # shift network to device
        self.net_to_device()
        if not cfg.is_test: self.init_optim(cfg)

    def init_optim(self, cfg):
        # optimizer and scheduler
        param_groups = self.grouping_params(cfg)
        # todo-note: momentum=0.9 is important for detection
        self.optimizer = {"det": torch.optim.SGD(param_groups['det'], lr=cfg.lr, momentum=0.9)}
        self.lr_scheduler = {"det": self.get_scheduler(
            self.optimizer['det'], cfg.lr_policy, cfg.lr_decay_milestones, cfg)
        }
        if not cfg.uni_optim:
            self.optimizer.update({"reid": torch.optim.Adam(param_groups['reid'])})
            lr_decay = [ceil(0.6 * cfg.max_epoch), ceil(0.8 * cfg.max_epoch)]
            self.lr_scheduler.update({"reid": self.get_scheduler(
                self.optimizer['reid'], cfg.lr_policy, lr_decay, cfg)
            })

    def set_input(self, input):
        # images: list of raw tensor img (3 h w), 0-255, raw img w/o normalization, BGR used for caffe pre-trained model
        # targets：list of dict with keys: 'gt_pids' 'boxes' 'gt_classes'
        images, targets = input['images'], input['targets']
        self.inp_img = list(img.to(self.device) for img in images)
        self.inp_tgt = [{k: self.to_device(v) for k, v in t.items()} for t in targets]

    def forward(self):
        detections, losses = self.detector(self.inp_img, self.inp_tgt)
        self.detections = detections
        self.reid_time = self.detector.reid_time
        return detections, losses

    def optimize_parameters(self, cfg):
        detections, losses = self.forward()
        # combine losses
        loss = sum([v.mean() for v in losses.values()])
        if torch.isnan(loss) or torch.isinf(loss):
            print("Nan or Inf loss occurred, ignore this batch.")
            return

        # backward
        self.detector.zero_grad()
        for optimizer in self.optimizer.values(): optimizer.zero_grad()
        loss.backward()
        # NOTE: clipping avoids NaN or Inf, but drops performance
        if cfg.clip_grad: torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 20.)
        for optimizer in self.optimizer.values(): optimizer.step()

        # record losses and classifier accuracy
        losses.update({"acc": sum([item['acc'] for item in self.detections]) / cfg.batch_size * 100})
        losses = self.decode_losses(losses)
        self.loss_meter.update(losses)

        return detections, losses

    def grouping_params(self, cfg):
        det_params, reid_params = [], []
        num_det_params, num_reid_params = 0, 0
        for key, value in dict(self.detector.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    det_params += [{'params': [value], 'lr': cfg.lr * (cfg.double_bias + 1),
                                    # 5e-4 if bias_decay else 0
                                    'weight_decay': cfg.bias_decay and cfg.weight_decay or 0}]
                    num_det_params += value.numel()
                elif "reid" in key:
                    reid_params += [{'params': [value], 'lr': cfg.reid_lr, 'weight_decay': cfg.weight_decay}]
                    num_reid_params += value.numel()
                    # print("name: {}, #params: {:.2f}".format(key, value.numel() / 1e6))
                else:
                    det_params += [{'params': [value], 'lr': cfg.lr, 'weight_decay': cfg.weight_decay}]
                    num_det_params += value.numel()
                    # print("name: {}, #params: {:.2f}".format(key, value.numel() / 1e6))
        if cfg.uni_optim:
            det_params.extend(reid_params)
            self.logger.msg("* #ModelParams: {:.2f}M".format((num_det_params + num_reid_params) / 1e6))
            return {"det": det_params}
        else:
            # separating parameters of detection and reid.
            self.logger.msg("* #DetParams: {:.2f}M #reIDParams: {:.2f}M".format(num_det_params / 1e6, num_reid_params / 1e6))
            return {"det": det_params, "reid": reid_params}

    def extra_box_feat(self, image, target):
        box_feat = self.detector.extra_box_feat(image, target)
        return box_feat

    def format_iter_msg(self, epoch, epoch_iter, n_batch, t):
        fg_cnt = sum([item['fg_cnt'] for item in self.detections])
        bg_cnt = sum([item['bg_cnt'] for item in self.detections])
        pk = sum([item['boxes'].size(0) for item in self.detections])
        msg = "epoch: %1d/%1d (%04d/%4d), time: %.3f, fg/bg: (%d/%d), pk: %d" \
              % (epoch, self.cfg.max_epoch - 1, epoch_iter, n_batch, t, fg_cnt, bg_cnt, pk)
        return msg

    @torch.no_grad()
    def get_current_visuals(self, save_path=None):
        # preparing
        images = [r['img'] for r in self.detections]
        targets = [r['tgt'] for r in self.detections]
        # list of (n*6), [x0,y0,x1,y1,lbl,pid]
        roi_lbl_pid = [torch.cat([t['boxes'], t['labels'][:, None].float(), t['pids'][:, None].float()], dim=1)
                       for t in self.detections]
        # plot boxes in image
        im_with_roi = vis.vis_roi_boxes(images, roi_lbl_pid, self.cfg.pixel_means, BGR=True)
        im_with_gt = vis.vis_gt_boxes(images, targets, self.cfg.pixel_means, BGR=True)
        im2show = torch.cat([im_with_roi, im_with_gt], dim=0)

        use_mask = False
        if use_mask:
            gtmasks = util.vis_masks([t['masks'] for t in targets])
            predmasks = util.vis_masks([t['mask'] for t in self.detections])
            im2show = torch.cat([im2show, predmasks, gtmasks], dim=0)

        im2show = im2show.index_select(dim=1, index=torch.tensor([2, 1, 0]))  # NCHW, BGR to RGB for vis
        im2show = vutils.make_grid(im2show, nrow=len(images), normalize=True, scale_each=True)  # CHW
        if save_path is not None:
            util.mkdir(os.path.dirname(save_path))
            imsave(save_path, im2show.detach().permute(1, 2, 0).numpy())
        visuals = dict(im_roi_gt=im2show)
        return visuals

    def init_DetHead(self, cfg):
        return
