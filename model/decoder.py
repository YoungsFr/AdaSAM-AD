import torch
import torch.nn as nn
import torch.nn.functional as F
from .DC import DeformableConv

class SegHead(nn.Module):
    def __init__(self, dim):
        super(SegHead, self).__init__()
        self.conv1 = DeformableConv(dim,dim,3,1,1,edge=False)
        self.conv2 = nn.Conv2d(dim,dim,3,1,1)
        self.BN = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        self.mask_conv = nn.Conv2d(dim,1,3,1,1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.BN(x)
        x = self.activation(x)
        mask = self.mask_conv(x)
        return mask

class HPPF(nn.Module):
    def __init__(self, dim, H, prompt_channels=72):
        super(HPPF, self).__init__()
        self.conv3_1 = nn.Conv2d(dim*4, dim, 1, 1, 0)          # 用于 x3 (288→72)
        self.conv2_1 = nn.Conv2d(dim*2, dim, 1, 1, 0)          # 用于 x2 (144→72)
        self.conv_prompt = nn.Conv2d(prompt_channels, dim, 1, 1, 0)  # 用于频率提示
        self.conv1_1 = nn.Conv2d(dim*3, dim, 1, 1, 0)          # 拼接后 (72*4 → 72)
        self.GAP = nn.AdaptiveAvgPool2d(H)
        self.GMP = nn.AdaptiveMaxPool2d(H)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, 1, 3, 1, 1)

    def forward(self, x1, x2, x3, prompt):
        # 处理 x2, x3
        x2 = self.conv2_1(x2)
        x2 = F.interpolate(x2, x1.shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.conv3_1(x3)
        x3 = F.interpolate(x3, x1.shape[-2:], mode='bilinear', align_corners=False)
        
        # 处理频率提示
        prompt = self.conv_prompt(prompt)                     # 通道对齐
        prompt = F.interpolate(prompt, x1.shape[-2:], mode='bilinear', align_corners=False)
        
        x1_cat = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1_1(x1_cat)
        x = x * prompt
        residual = x
        x_1 = self.GAP(x)
        x_2 = self.GMP(x)
        x = x_1 + x_2
        x = self.sigmoid(x)
        x = x * residual
        x = self.conv(x)                                       # [B, 1, H, W]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 恢复至原图
        return x





class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, first_layer=False, last_layer=False):
        super(Decoder, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.seg_head = SegHead(in_dim)

        self.conv = nn.Conv2d(in_dim*2,in_dim,3,1,1)

        self.last_conv = nn.Conv2d(72,3,3,1,1)
    
    def forward(self, x, y):
        if self.last_layer:
            x, mask = self.seg_head(x)
            return x, mask
        if self.first_layer:
            y = F.interpolate(y,scale_factor=2,mode='bilinear',align_corners=False)
            y = self.last_conv(y)
            x = self.conv(torch.concat([x,y],dim=1))
            x, mask = self.seg_head(x, self.first_layer)
            return x, mask
        x = self.conv(torch.concat([x,y],dim=1))
        x, mask = self.seg_head(x)
        return x, mask



