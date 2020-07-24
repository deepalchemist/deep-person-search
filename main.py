import os
import sys
import time
import pickle
import shutil
import inspect
import datetime
import numpy as np
from math import ceil
from scipy.misc import imsave

import torch
import torchvision.utils as vutils
import torch.utils.data as data

from lib.cfg.config import Config
import lib.misc.util as util
from lib.misc.sampler import collate_fn
from lib.misc.logger import Logger
import lib.data.getitem as getitem
from lib.data.preproc import create_data
from lib.engine.getengine import create_engine


def train_one_epoch(engine, data_loader, logger, epoch, warm_up=-1):
    engine.train()
    logger.msg('learning rate of epoch {}: {}'.format(
        epoch, [o.param_groups[0]['lr'] for o in engine.optimizer.values()]))

    n_batch = len(data_loader)
    epoch_iter = 0
    epoch_start_time = time.time()
    for idx, data in enumerate(data_loader):
        iter_start_time = time.time()
        logger.reset()
        epoch_iter += 1
        total_iter = epoch * n_batch + epoch_iter

        if warm_up > 0 and epoch < warm_up:
            engine.update_group_lr(epoch, total_iter)

        engine.set_input(data)
        _, losses = engine.optimize_parameters(cfg)

        # print something
        if total_iter % 100 == 0:
            # Time that single image cost at the training stage
            t = (time.time() - iter_start_time) / cfg.batch_size
            iter_msg = engine.format_iter_msg(epoch, epoch_iter, n_batch, t)
            logger.msg(iter_msg)
            # loss messages
            logger.print_current_losses(suffix='\t', losses=engine.loss_meter.avg)

        # display images and loss curve
        if cfg.display and total_iter % cfg.display_freq == 0:
            logger.plot_current_losses(losses, total_iter)
            # vis gt and roi boxes
            # with torch.no_grad():
            #     im_roi_gt = engine.get_current_visuals()
            # logger.display_current_results(im_roi_gt, total_iter)

    logger.msg('End of epoch %d / %d \t Time Taken: %.1f min' %
               (epoch, cfg.max_epoch - 1, (time.time() - epoch_start_time) / 60))
    torch.cuda.empty_cache()


def main(cfg):
    # --------------------------------------------------------------------------------------------------------------
    ### Initialize seed ###
    util.init_random_seed(cfg.seed)

    ### Initialize database ###
    imdb = create_data(cfg.benchmark, cfg.data_root, cfg.no_flip, training=True, use_mask=cfg.use_mask)
    roidb, num_class = imdb.roidb, imdb.num_classes
    dataset = getitem.SequentialGetitem(roidb, num_class, div=cfg.div, BGR=not cfg.RGB)

    train_sampler = torch.utils.data.RandomSampler(dataset)  # shuffled
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfg.batch_size, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler,
        num_workers=cfg.n_worker, collate_fn=collate_fn())
    cfg.n_batch_per_epoch = len(data_loader)

    ### Visualization and print messages ###
    logger = Logger(cfg.expr_dir, cfg.display)
    cfg.logger = logger  # logger as engine attribute

    ### Initialize engine and model ###
    engine = create_engine(cfg)
    # load checkpoint if test or continue training
    engine.setup(cfg)

    # calculate flops and #params # TODO
    # total_ops, total_params = util.get_model_complexity(detector, device)
    # print("* #TotalParams: %.2fM Flops: %.2f" % (total_params, total_ops))

    logger.msg("* Experiment name: {}\n".format(cfg.expr_name))
    # --------------------------------------------------------------------------------------------------------------
    # Start training
    start_time = time.time()
    for epoch in range(engine.start_epoch, cfg.max_epoch):
        engine.update_group_lr(epoch)
        train_one_epoch(engine, data_loader, logger, epoch, cfg.warmup_epochs)
        engine.save_checkpoint(epoch=epoch)
        # evaluation
        if epoch in cfg.eval_epoch or epoch == cfg.max_epoch - 1:
            print('evaluating ...')
            cfg.load_ckpt = os.path.join(cfg.expr_dir, 'latest.pth')
            with torch.no_grad():
                evaluate(cfg)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    logger.msg("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    logger.msg("Experiment name: {}".format(cfg.expr_name))


@torch.no_grad()
def evaluate(cfg):
    print("Experiment dir: {}".format(cfg.expr_dir))
    util.init_random_seed(cfg.seed)
    # device = torch.device(cfg.device)

    ### Initialize database ###
    imdb = create_data(cfg.benchmark, cfg.data_root, True, training=False, probe_type=cfg.probe_type)
    roidb = imdb.roidb
    print('{:d} roidb entries'.format(len(roidb)))  # number of training/test images
    probe_getitem = getitem.ProbeGetitem(imdb.probes)
    gallery_getitem = getitem.SequentialGetitem(roidb, imdb.num_classes, div=cfg.div, BGR=not cfg.RGB)
    probe_loader = torch.utils.data.DataLoader(probe_getitem, batch_size=1,
                                               num_workers=cfg.n_worker, collate_fn=collate_fn())
    gallery_loader = torch.utils.data.DataLoader(gallery_getitem, batch_size=cfg.eval_batch_size,
                                                 num_workers=cfg.n_worker, collate_fn=collate_fn())

    # logger
    logger = Logger(cfg.expr_dir, cfg.display)
    cfg.logger = logger  # logger as engine attribute

    ### Initialize engine and model ###
    engine = create_engine(cfg)  # logger as engine attribute
    # load checkpoint if test or continue training
    engine.setup(cfg)
    engine.eval()  # NOTE: important

    # dir for saving features and detections
    output_dir = os.path.join(imdb.cache_path, imdb.name, 'detections')  # ./cache/benchmark_name/features
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----------------------------------------------------------------------------------------------------------------
    # Traversing gallery images
    def extract_gallery():
        num_images = len(imdb.image_names)
        gallery_boxes = []  # len(gallery_boxes)=num_classes
        gallery_features = []

        _t = {'im_detect': util.Timer(), 'reid': util.Timer()}

        print("\nextracting gallery features ... ")
        for idx, data in enumerate(gallery_loader):
            engine.set_input(data)
            torch.cuda.synchronize()

            _t['im_detect'].tic()
            rois, _ = engine.forward()
            _t['im_detect'].toc()
            _t['reid'].duration(engine.reid_time)

            for roi in rois:
                box_and_prob = torch.cat([roi['boxes'], roi['scores'][:, None]], dim=1)
                gallery_boxes.append(box_and_prob.cpu().numpy())
                gallery_features.append(roi['feats'].cpu().numpy())

            # if cfg.display:
            #     im = cv2.imread(imdb.image_path_at(i))
            #     im2show = np.copy(im)
            # if cfg.display:
            #     im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.0)
            # misc_tic = time.time()
            # misc_toc = time.time()
            # nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(
                idx + 1, len(gallery_loader) + 1, _t['im_detect'].average_time, _t['reid'].average_time))
            sys.stdout.flush()

            # if cfg.display:
            #     cv2.imwrite(os.path.join(output_dir, 'gallery.png'), im2show)

        print('total time:{:.3f}s, reid time:{:.3f}s'.format(_t['im_detect'].average_time, _t['reid'].average_time))

        assert len(gallery_features) == len(gallery_boxes) == num_images

        with open(os.path.join(output_dir, 'gallery_features.pkl'), 'wb') as f:
            pickle.dump(gallery_features, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(output_dir, 'gallery_boxes.pkl'), 'wb') as f:
            pickle.dump(gallery_boxes, f, pickle.HIGHEST_PROTOCOL)
        return gallery_boxes, gallery_features

    # ----------------------------------------------------------------------------------------------------------------
    # Traversing probe images
    def extract_probe():
        num_probe_ims = len(imdb.probes)
        probe_features = {'feat': [0 for _ in range(num_probe_ims)]}
        _t = {'im_exfeat': util.Timer(), 'misc': util.Timer()}

        print("\nextracting probe features ... ")
        for i, data in enumerate(probe_loader):
            images, target = data['images'], data['targets']
            image = list(img.to(engine.device) for img in images)
            target = [{"boxes": t.view(-1, 4).to(engine.device)} for t in target]
            torch.cuda.synchronize()
            assert len(image) == 1, "Only support single image input"

            _t['im_exfeat'].tic()
            feat = engine.extra_box_feat(image, target)
            _t['im_exfeat'].toc()

            probe_features['feat'][i] = feat[0].cpu().numpy()

            # print('im_exfeat: {:d}/{:d} {:.3f}s'.format(i + 1, num_probe_ims, _t['im_exfeat'].average_time))
            sys.stdout.write('im_exfeat: {:d}/{:d} {:.3f}s   \r' \
                             .format(i + 1, num_probe_ims, _t['im_exfeat'].average_time))
            sys.stdout.flush()

            # if cfg.display:
            #     im2show = vis_detections(np.copy(im), 'person', np.concatenate((roi, np.array([[1.0]])), axis=1), 0.3)
            #     cv2.imwrite(os.path.join(output_dir, 'probe.png'), im2show)

        with open(os.path.join(output_dir, 'probe_features.pkl'), 'wb') as f:
            pickle.dump(probe_features, f, pickle.HIGHEST_PROTOCOL)
        return probe_features

    # ----------------------------------------------------------------------------------------------------------------
    start = time.time()
    flag = False
    if flag:
        # load pickle file for debugging evaluation code
        with open(os.path.join(output_dir, 'probe_features.pkl'), 'rb') as f:
            probe_features = pickle.load(f)
        with open(os.path.join(output_dir, 'gallery_features.pkl'), 'rb') as f:
            gallery_features = pickle.load(f)
        with open(os.path.join(output_dir, 'gallery_boxes.pkl'), 'rb') as f:
            gallery_boxes = pickle.load(f)
    else:
        with torch.no_grad():
            probe_features = extract_probe()
            gallery_boxes, gallery_features = extract_gallery()

    # ----------------------------------------------------------------------------------------------------------------
    # evaluate pedestrian detection and person search
    logger.msg('\nevaluating detections')

    # all detection
    # todo-bug: det_thr 0.5 to 0.05 for single-stage detector
    ap, recall = imdb.evaluate_detections(gallery_boxes, score_thresh=cfg.det_thr)
    logger.msg('all detection:')
    logger.msg('  recall = {:.2%}'.format(recall))
    logger.msg('  ap = {:.2%}'.format(ap))

    # labeled only detection
    ap, recall = imdb.evaluate_detections(gallery_boxes, score_thresh=cfg.det_thr, labeled_only=True)
    logger.msg('labeled_only detection:')
    logger.msg('  recall = {:.2%}'.format(recall))  # todo(note): added 20200609
    logger.msg('  ap = {:.2%}'.format(ap))

    # evaluate search
    kwargs = dict(score_thresh=0.5, base_iou_thresh=0.5, dump_json=os.path.join(cfg.expr_dir, 'results.json'))
    if "ssm" == cfg.benchmark: kwargs.update({"gallery_size": cfg.gallery_size})
    ap, topk, accs, recall = imdb.evaluate_search(gallery_boxes, gallery_features, probe_features['feat'], **kwargs)

    logger.msg('search ranking:')
    logger.msg('  recall = {:.2%}'.format(recall))  # does the true_positive of a probe been detected.
    logger.msg('  mAP = {:.2%}'.format(ap))
    for i, k in enumerate(topk):
        logger.msg('  top-{:2d} = {:.2%}'.format(k, accs[i]))

    end = time.time()
    print("test time: %0.1fmin" % ((end - start) / 60))

if __name__ == '__main__':
    cfg = Config().parse()
    if cfg.is_test:
        with torch.no_grad():
            evaluate(cfg)
    else:
        shutil.copy(inspect.getfile(main), cfg.expr_dir)
        main(cfg)
