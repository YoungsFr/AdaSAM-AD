import os
import time
import random
import logging
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

def binarize_threshold(tensor, threshold=0.5):
    return (tensor > threshold).int() * 255

def set_seed(seed, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, logger):
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found at {}".format(checkpoint_path))
        return 0, float('inf')
    
    logger.info("Loading checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_dice = checkpoint['best_dice']
    best_iou = checkpoint['best_iou']
    logger.info(f"Resumed from epoch {start_epoch-1}")
    
    return start_epoch, best_iou, best_dice

def save_checkpoint(epoch, model, optimizer, scheduler, best_iou, best_dice, is_best, save_dir, logger, filename='checkpoint.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'best_iou': best_iou,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    filepath = os.path.join(save_dir, f'epoch{epoch}.pth')
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved at epoch {epoch}")
    