import torch
import numpy as np

# From https://github.com/OpenGVLab/SAM-Med2D/blob/main/metrics.py

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def mae(pr, gt):
    pr_, gt_ = _list_tensor(pr, gt)
    mae_val = torch.mean(torch.abs(pr_ - gt_), dim=[1, 2, 3])
    return mae_val.cpu().numpy()

def e_measure(pr, gt, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    
    b = pr_.shape[0]
    e_vals = []
    
    for i in range(b):
        p = pr_[i]
        g = gt_[i]
        
        if torch.sum(g) == 0: 
            e_val = 1.0 - torch.mean(p).item()
        elif torch.sum(g) == g.numel():
            e_val = torch.mean(p).item()
        else:
            mean_p = torch.mean(p)
            mean_g = torch.mean(g)

            p_c = p - mean_p
            g_c = g - mean_g
            
            align_matrix = 2 * (p_c * g_c) / (p_c**2 + g_c**2 + 1e-8)
            
            enhanced = ((align_matrix + 1) ** 2) / 4
            e_val = torch.mean(enhanced).item()
            
        e_vals.append(e_val)
        
    return np.array(e_vals)


def calculate_mae(pred, gt):
    
    if pred.shape != gt.shape:
        pred = torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)


    mae = torch.mean(torch.abs(pred - gt))
    
    return mae.item()

def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'mae':
            metric_list.append(np.mean(mae(pred, label)))
        elif metric == 'me':
            metric_list.append(np.mean(e_measure(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric