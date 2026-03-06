import os
import os.path as osp
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seeds(_seed_):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    # torch.use_deterministic_algorithms(True)

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)
    return

class WarmUp():
    def __init__(self, ed_it, st_lr, ed_lr):
        self.ed_it = ed_it
        self.st_lr = st_lr
        self.ed_lr = ed_lr
    
    def get_lr(self, cur_it):
        return self.st_lr + (self.ed_lr - self.st_lr) / self.ed_it * cur_it

    def adjust_lr(self, optimizer, cur_it):
        if cur_it <= self.ed_it:
            lr = self.get_lr(cur_it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        # val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
        #                 zip(self.names, self.val)])
        # avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
        #                 zip(self.names, self.avg)])
        out = ' '.join(['{} {:.{}f} ({:.{}f})'.format(n, v, self.precision, a, self.precision) for n, v, a in
                        zip(self.names, self.val, self.avg)])
        # return '{} ({})'.format(val, avg)
        return '{}'.format(out)
    

class InputPadder:
    """ Pads images such that dimensions are divisible by 16 """
    def __init__(self, dims, padsize=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padsize) + 1) * padsize - self.ht) % padsize
        pad_wd = (((self.wd // padsize) + 1) * padsize - self.wd) % padsize
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def supervised_loss(flow, flow_gt, gamma=0.8):
    n_predictions = len(flow)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        for j in range(len(flow_gt)):
            i_loss = (flow[i][j] - flow_gt[j]).abs()
            flow_loss += i_weight * i_loss.mean() / len(flow_gt)
    loss_deriv_dict = {}
    loss_deriv_dict['flow_mean'] = flow[-1][0].abs().mean()
    return flow_loss, loss_deriv_dict


def supervised_loss_two_point(flow_preds, flow_gt, gamma=0.8):
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss.mean()
        
    loss_deriv_dict = {}
    loss_deriv_dict['flow_mean'] = flow_preds[0].abs().mean()
    return flow_loss, loss_deriv_dict


def light_intensity_loss(pred_imgs, imgs_gt):
    light_loss = 0.0
    for img, gt in zip(pred_imgs, imgs_gt):
        light_loss += (img - gt).abs().mean() / len(pred_imgs)
    return light_loss
##########################################################################################################
## Compute Error

def get_class_metric(metric_dict, len_dict):
    CLASS_A = [
        '2016-09-02_000170',
        'car_tuebingen_000024',
        'car_tuebingen_000145',
        'motocross_000108']
    CLASS_B = [
        'ball_000000',
        'kids__000002',
        'skatepark_000034',
        'spitzberglauf_000009']
    CLASS_C = [
        'car_tuebingen_000103',
        'horses__000028',
        'tubingen_05_09_000012']
    
    a_metric_sum = 0
    b_metric_sum = 0
    c_metric_sum = 0
    a_len_sum = 0
    b_len_sum = 0
    c_len_sum = 0
    for k, v in metric_dict.items():
        if k in CLASS_A:
            a_metric_sum += v * len_dict[k]
            a_len_sum += len_dict[k]
        elif k in CLASS_B:
            b_metric_sum += v * len_dict[k]
            b_len_sum += len_dict[k]
        elif k in CLASS_C:
            c_metric_sum += v * len_dict[k]
            c_len_sum += len_dict[k]
    
    return a_metric_sum/a_len_sum, b_metric_sum/b_len_sum, c_metric_sum/c_len_sum


def calculate_error_rate(pred_flow, gt_flow):
    pred_flow = pred_flow.squeeze(dim=0).permute([1,2,0]).cpu().numpy()
    gt_flow = gt_flow.squeeze(dim=0).permute([1,2,0]).cpu().numpy()

    epe_map = np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2))
    bad_pixels = np.logical_and(
        epe_map > 0.5,
        epe_map / np.maximum(
            np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
    return bad_pixels.mean() * 100.



    
##########################################################################################################
## Flow Viz

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=True):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def vis_flow_batch(forw_flows_list, save_path, suffix='forw_flow', max_batch=4):
    # forward flow
    for ii in range(forw_flows_list[0].shape[0]):
        for jj in range(len(forw_flows_list)):
            forw_flows = forw_flows_list[jj]
            flow = forw_flows[ii, :]
            flow = flow.permute([1, 2, 0]).detach().cpu().numpy()
            if jj == 0:
                flow_viz = flow_to_image(flow, convert_to_bgr=True)
            else:
                flow_viz = np.concatenate([flow_viz, flow_to_image(flow, convert_to_bgr=True)], axis=1)        
        if ii == 0:
            flow_viz_all = flow_viz
        else:
            flow_viz_all = np.concatenate([flow_viz_all, flow_viz], axis=0)
        if ii-1 >= max_batch:
            break
    cur_save_path = osp.join(save_path, '{:s}.png'.format(suffix))
    cv2.imwrite(cur_save_path, flow_viz_all)


def vis_img_batch(imgs, save_path, max_batch=4):
    # img
    for ii in range(imgs.shape[0]):
        img = imgs[ii, :]
        img = img.permute([1, 2, 0]).detach().cpu().numpy() * 255.
        cur_save_path = osp.join(save_path, 'img_batch{:d}.png'.format(ii))
        cv2.imwrite(cur_save_path, img.astype(np.uint8))
        if ii-1 >= max_batch:
            break


def vis_flow_batch_two_point(forw_flows, save_path, suffix='forw_flow', max_batch=4):
    # forward flow
    for ii in range(len(forw_flows)):
        flow = forw_flows[ii, :]
        flow = flow.permute([1, 2, 0]).detach().cpu().numpy()
        if ii == 0:
            flow_viz = flow_to_image(flow, convert_to_bgr=True)
        else:
            flow_viz = np.concatenate([flow_viz, flow_to_image(flow, convert_to_bgr=True)], axis=1)        
    cur_save_path = osp.join(save_path, '{:s}.png'.format(suffix))
    cv2.imwrite(cur_save_path, flow_viz)
