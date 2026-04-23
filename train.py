import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import random
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR,CosineAnnealingWarmRestarts
import cv2
import numpy as np

from model.SAM import AdaSAM
from sam2.build_sam import build_sam2
from dataset import get_train_val_loaders
from loss import cal_loss
from metrics import SegMetrics
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile

def setup_log_file(args):
    path = '.experiments'
    if not os.path.exists(path):
        os.makedirs(path)
    
    log_file_path = os.path.join(path, f"{args.exp}_log.txt")
    
    if not args.resume:
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"experiment: {args.exp}\n")
            f.write(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            # f.write(f"description: {description}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epoch':<8} | {'Loss':<10} | {'Train_Acc':<10} | {'mIoU':<10} | {'mDice':<10}\n")
            f.write("-" * 80 + "\n")
    return log_file_path

def write_log(file_path, epoch, loss, acc, miou, mdice):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{epoch+1} | {loss} | {acc}% | {miou} | {mdice}\n")

def set_seed(seed, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_checkpoint(args, epoch, model, optimizer, scheduler):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    filepath = os.path.join(args.checkpoint_path, f'epoch{epoch + 1}.pth')
    torch.save(state, filepath)
    logger.info(f"checkpoint{epoch + 1} saved at: {filepath}")

def load_checkpoint(args, model, optimizer, scheduler, device):
    checkpoint_path = os.path.join(args.checkpoint_path, args.exp)
    assert os.path.exists(checkpoint_path)
    count = len(os.listdir(checkpoint_path))
    if count == 0:
        return 0
    pth = os.path.join(checkpoint_path,f'epoch{count}.pth')
    
    logger.info(f"Loading latest checkpoint")
    checkpoint = torch.load(pth, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()

    total_loss = torch.tensor(0.0).to(device)
    total_correct = torch.tensor(0).to(device)
    total_pixels = torch.tensor(0).to(device)
    total_samples = torch.tensor(0).to(device)
    
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=True)
    for image, mask, _ in progress_bar:
        image, mask_gt = image.to(device), mask.to(device)
        
        optimizer.zero_grad()
        masks = model(image)
        total_bce, total_dice = cal_loss(masks, mask_gt)
        loss = total_bce + total_dice
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        with torch.no_grad():
            pred_mask = masks[0] if isinstance(masks, list) else masks
            predictions = (torch.sigmoid(pred_mask) > 0.5).float()
            correct = (predictions == mask_gt).sum()
            total_correct += correct
            total_pixels += mask_gt.numel()
        
        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        current_acc = 100.0 * total_correct.item() / total_pixels.item()
        
        progress_bar.set_postfix(
            Dice=f'{total_dice:.4f}',
            BCE=f'{total_bce:.4f}',
            loss=f'{avg_loss:.4f}', 
            acc=f'{current_acc:.2f}%'
        )
    epoch_acc = 100.0 * total_correct.item() / total_pixels.item()
    return avg_loss, epoch_acc

@torch.no_grad()
def validate(args, model, dataloader, device, epoch):
    model.eval()
    count = 0
    iou_total = 0
    dice_total = 0
    progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch+1}', leave=True)

    for i, (image, mask, _) in enumerate(progress_bar):
        image = image.to(device)
        gt = mask.to(device)
        preds = model(image)
        iou = SegMetrics(preds[0], gt, metrics='iou')
        dice = SegMetrics(preds[0], gt, metrics='dice')
        iou_total += iou
        dice_total += dice
        count += 1

    miou = iou_total / count
    mdice = dice_total / count
    logger.info(f"Val Epoch {epoch+1} - mIoU: {miou.item():.4f}, mDice: {mdice.item():.4f}")
    return miou, mdice

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed, logger)
    log_file_path = setup_log_file(args)
    
    """ Config model """
    sam2_model = build_sam2(args.sam_cfg_path, args.sam_path, device).eval()

    for param in sam2_model.parameters():
        param.requires_grad = False

    model = AdaSAM(sam2_model=sam2_model).to(device)

    """ Load Data """
    train_loader, val_loader = get_train_val_loaders(args.data_path,args.val_path,img_size=args.img_size, batch_size=args.batch_size)
    
    """ Config optimizer """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    
    """ Config scheduler """
    if args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, threshold=1e-4, min_lr=1e-6,cooldown=2)
    elif args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    else:
        scheduler = None

    """ checkpoints """
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler, device)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        miou, mdice = validate(args, model, val_loader, device, epoch)

        scheduler.step()

        save_checkpoint(args, epoch, model, optimizer, scheduler)

        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")


    
    logger.info("Training completed")



def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    """experiment configuration"""
    parser.add_argument('--exp', type=str, default='2',
                        help='name of the experiment')
    parser.add_argument('--data_path', type=str, default='/root/datasets/PolypDataset/train',
                        help='path of dataset')
    parser.add_argument('--val_path', type=str, default='/root/datasets/PolypDataset/val')
    parser.add_argument('--checkpoint_path', type=str, default='/workspace/AdaSAM-AD/checkpoints/Polyp',
                        help='directory to save checkpoints')
    

    """ model configuration """
    parser.add_argument('--sam_path', type=str, default='./sam2_hiera_large.pt',
                        help='path of sam2 model')
    parser.add_argument('--sam_cfg_path', type=str, default='sam2_hiera_l.yaml')
    parser.add_argument('--img_size', type=int, default=224)

    """ training configuration """
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=24, 
                        help='set batch size')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of total epochs to run')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='leraning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine', 'none'],
                        help='learning rate scheduler')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD','Adam','AdamW','RMSprop'],
                        help='choose optimizer')


    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main() 