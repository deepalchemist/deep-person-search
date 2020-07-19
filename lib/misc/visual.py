import cv2
import torch


def vis_roi_boxes(im_batch, roi_lbl_pid, pixel_means, BGR=True):
    ''' Visualize a mini-batch for debugging.

    Args:
        im_batch (list of 3D tensor): (3 H W) BGR
        roi_lbl_pid (list of 2D tensor): (bs MAX_NUM_ROI_BOXES 6) consist of reg, detect_prob, pid_pred
        BGR: color order of images in im_batch
    '''
    batch_size = len(im_batch)
    pixel_means = torch.tensor(pixel_means, dtype=torch.float)[:, None, None]
    batch_cv2im = []
    for ii in range(batch_size):
        rec_im = im_batch[ii].detach().cpu() + pixel_means  # CHW
        if rec_im.max().item() <= 1.: rec_im = rec_im * 255.
        if not BGR:
            rec_im = rec_im.index_select(dim=0, index=torch.tensor([2, 1, 0]))  # CHW, RGB to BGR

        cv2im = rec_im.permute(1, 2, 0).numpy()  # HWC
        for x1, y1, x2, y2, is_person, pid in roi_lbl_pid[ii]:
            if pid > -1:  # id-labeled
                ec = (0, 255, 0)  # (green)
            elif pid == -1 and is_person == 1:  # is-human-wo-id
                ec = (255, 0, 0)  # (blue)
            else:
                ec = (0, 0, 255)  # (red)
            if pid > -1 or is_person == 1:
                # thickness=negative means filled rectangle
                cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec, thickness=-1)
                # cv2im = cv2.rectangle(cv2im, (x1, y2), (x2, y2 + 16), ec, thickness=-1)  # thickness=negative means filled rectangle
                cv2im = cv2.putText(cv2im, "{}&{}".format(str(int(pid)), str(int(is_person))), org=(x1 + 4, y1 - 2),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
            cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)

        batch_cv2im.append(cv2im)
    # .get() convert cv2.UMat to numpy array
    batch_tensor_im = [torch.as_tensor(im.get()).permute(2, 0, 1) for im in batch_cv2im]  # CHW
    ret = torch.stack(batch_tensor_im, dim=0)

    # batching images with diff size
    # ret = batching_images_by_padding(batch_tensor_im)
    return ret


def vis_gt_boxes(im_batch, target, pixel_mean, BGR=True, query_pid=None):
    ''' Visualize a mini-batch for debugging.

    Args:
        im_batch (list of 3D tensor): (3 H W) BGR
        target (list of dict): (bs MAX_NUM_GT_BOXES 6)
        BGR: color order of images in im_batch
    '''
    batch_size = len(im_batch)
    pixel_means = torch.tensor(pixel_mean, dtype=torch.float)[:, None, None]
    batch_cv2im = []
    for ii in range(batch_size):
        rec_im = im_batch[ii].detach().cpu() + pixel_means
        if rec_im.max().item() <= 1.: rec_im = rec_im * 255.

        if not BGR:
            rec_im = rec_im.index_select(dim=0, index=torch.tensor([2, 1, 0]))  # CHW, RGB to BGR
        cv2im = rec_im.permute(1, 2, 0).numpy()
        for (x1, y1, x2, y2), is_person, pid in zip(target[ii]['boxes'], target[ii]['labels'], target[ii]['pids'], ):
            if is_person != 1:
                continue
            ec = (255, 0, 0) if pid == -1 else (0, 255, 0)
            if pid != -1:
                # thickness=negative means filled rectangle
                cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec, thickness=-1)
                cv2im = cv2.putText(cv2im, str(int(pid)), org=(x1 + 4, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)
        # print query pid
        if query_pid is not None:
            cv2im = cv2.putText(cv2im, str(int(query_pid[ii])), org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1., color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        batch_cv2im.append(cv2im)

    batch_tensor_im = [torch.as_tensor(im.get()).permute(2, 0, 1) for im in batch_cv2im]
    ret = torch.stack(batch_tensor_im, dim=0)

    # batching images with diff size
    # ret = batching_images_by_padding(batch_tensor_im)
    return ret
