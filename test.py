import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import logging
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.SAM import AdaSAM
from sam2.build_sam import build_sam2

from dataset import get_test_loader
from metrics import SegMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def save_visualization(image, mask_gt, mask_pred_raw, save_path, name):
    # 1. 基础图像处理
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255
    img = img.astype(np.uint8)
    
    # 2. 获取概率图和二值预测图
    prob_map = torch.sigmoid(mask_pred_raw).squeeze().cpu().numpy()
    gt = (mask_gt.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    pred_binary = (prob_map > 0.5).astype(np.uint8)
    
    # 3. 计算“认反了”的像素 (对应分布图两端的重叠区)
    # FN (红色分布在0附近的像素): 真实是息肉(1)，但模型打分极低(<0.1)
    fn_mask = (gt == 1) & (prob_map < 0.1)
    # FP (蓝色分布在1附近的像素): 真实是背景(0)，但模型打分极高(>0.9)
    fp_mask = (gt == 0) & (prob_map > 0.9)

    # 4. 生成误差诊断图 (Error Diagnosis)
    # 在原图基础上，将 FN 涂为蓝色，FP 涂为绿色
    error_diag = img.copy()
    error_diag[fn_mask] = [0, 0, 255]  # 漏诊 (Blue) - BGR格式下OpenCV保存
    error_diag[fp_mask] = [0, 255, 0]  # 误报 (Green)

    # 5. 准备拼接
    gt_color = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred_binary * 255, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 统一转为BGR供cv2保存
    
    # 横向拼接: 原图 | 真值 | 预测图 | 误差诊断
    combined = np.hstack([img_bgr, gt_color, pred_color, cv2.cvtColor(error_diag, cv2.COLOR_RGB2BGR)])
    
    cv2.imwrite(os.path.join(save_path, f"{name}_diag.png"), pred_color)
    # print(name)




def test(args, data_path, sam2_model, index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AdaSAM(sam2_model=sam2_model).to(device)

    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        # logger.info(f"Successfully loaded checkpoint: {args.checkpoint_file}")
    else:
        # logger.error(f"Checkpoint not found: {args.checkpoint_file}")
        return 0, 0

    test_loader = get_test_loader(data_path, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)

    model.eval()
    all_iou = []
    all_dice = []
    all_mae = []
    all_me = []
    
    prob_samples_pos = []
    prob_samples_neg = []

    save_dir = os.path.join(args.output_dir, args.experiment_name, os.path.basename(data_path))
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Testing on dataset: {os.path.basename(data_path)}")
    with torch.no_grad():
        for i, (image, mask_gt, image_path) in enumerate(tqdm(test_loader)):

            torch.cuda.empty_cache()
            image = image.to(device)

            masks = model(image) 
            mask_gt = mask_gt.to(device)
            iou_score = SegMetrics(masks[0], mask_gt, metrics='iou')
            dice_score = SegMetrics(masks[0], mask_gt, metrics='dice')
            mae_score = SegMetrics(masks[0], mask_gt, metrics='mae')
            me_score = SegMetrics(masks[0], mask_gt, metrics='me')
            all_iou.append(iou_score)
            all_dice.append(dice_score)
            all_mae.append(mae_score)
            all_me.append(me_score)

            save_visualization(image[0], mask_gt[0], masks[0][0], save_dir, image_path)

    mean_iou = sum(all_iou) / len(all_iou)
    mean_dice = sum(all_dice) / len(all_dice)
    # mean_mae = sum(all_mae) / len(all_mae)
    # mean_me = sum(all_me) / len(all_me)
    # return mean_iou.item(), mean_dice.item(), mean_mae.item(), mean_me.item()
    return mean_iou.item(), mean_dice.item()

def test_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading Base Model...")
    
    sam2_model = build_sam2(args.sam_cfg_path, args.sam_path, device)
    sam2_model.eval()

    dataset_names = ['CVC-300', 'ClinicDB', 'ColonDB', 'ETIS', 'kvasir']
    # dataset_names = [
    #     'bottle',
    #     'cable',
    #     'capsule',
    #     'carpet',
    #     'grid',
    #     'hazelnut',
    #     'leather',
    #     'metal_nut',
    #     'pill',
    #     'screw',
    #     'tile',
    #     'toothbrush',
    #     'transistor',
    #     'wood',
    #     'zipper',
    # ]
    results = {}
    index = 1
    for name in dataset_names:
        data_path = os.path.join(args.data_path, name)
        if not os.path.exists(data_path):
            logger.warning(f"Path skipped (not found): {data_path}")
            continue
            
        # iou, dice, mae, me = test(args, data_path, sam2_model, index)
        iou, dice = test(args, data_path, sam2_model, index)
        index += 1
        results[name] = [iou, dice]


    print("\n" + "="*50)
    print(f"{'Dataset':<20} | {'mIoU':<10} | {'mDice':<10} | {'MAE':<10} | {'mE':<10}")
    print("-" * 50)
    for key, val in results.items():
        # print(f"{key:<20} | {val[0]:.4f}     | {val[1]:.4f}     | {val[2]:.4f}     | {val[3]:.4f}")
        print(f"{key:<20} | {val[0]:.4f}     | {val[1]:.4f}     ")
    print("="*50)
    return results

def save_results_to_txt(results, epoch, file_path="eval_results.txt"):
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f'epoch{epoch}\n')
        f.write(f"{'Dataset':<20} | {'mIoU':<10} | {'mDice':<10} | {'MAE':<10} | {'mE':<10}\n")
        f.write(f"{'-'*48}\n")
        

        for key in results.keys():
            # line = f"{key:<20} | {results[key][0]:<10.4f} | {results[key][1]:<10.4f} | {results[key][2]:<10.4f} | {results[key][3]:<10.4f}\n"
            line = f"{key:<20} | {results[key][0]:<10.4f} | {results[key][1]:<10.4f} \n"
            f.write(line)
        f.write(f"{'='*50}\n\n\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='indus')
    parser.add_argument('--data_path', type=str, default='/root/datasets/PolypDataset/test')
    parser.add_argument('--checkpoint_file', type=str, default='/workspace/AdaSAM-AD/checkpoints/Polyp/epoch140.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    
    parser.add_argument('--sam_path', type=str, default='./sam2_hiera_large.pt')
    parser.add_argument('--sam_cfg_path', type=str, default='sam2_hiera_l.yaml')
    
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=224)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    results = test_main(args)
    print(args.checkpoint_file)
    save_results_to_txt(results, 130, f'./results/result.txt')


