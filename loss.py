import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    
    intersection = (preds * targets).sum(dim=1)
    dice = (2. * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return (1 - dice).mean()


def cal_loss(preds, gt):
    total_bce = 0
    total_dice = 0
    total_focal = 0
    weights = [1.0,0.5,0.3,0.2,0.1]
    sum_weights = sum(weights[:len(preds)])
    target_size = gt.shape[-2:]
    for i in range(0, len(preds)):
        if preds[i].shape[-2:] != target_size:
            pred_aligned = F.interpolate(preds[i], size=target_size, mode='bilinear', align_corners=False)
        else:
            pred_aligned = preds[i]

        loss_bce = F.binary_cross_entropy_with_logits(pred_aligned, gt)
        loss_dice = dice_loss(pred_aligned, gt)
        
        total_bce += loss_bce * weights[i]
        total_dice += loss_dice * weights[i]
    total_bce /= sum_weights
    total_dice /= sum_weights
    return total_bce, total_dice