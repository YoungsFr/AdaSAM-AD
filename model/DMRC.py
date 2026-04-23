from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from .DC import DeformableConv


class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, h, w = x.shape     # shape = [32, 64, 2000, 80]

        y = self.avg_pool(x)      # shape = [32, 64, 1, 1]
        y = y.view(b, c)                # shape = [32,64]

        y = self.linear1(y)             # shape = [32, 64] * [64, 4] = [32, 4]

        y = self.linear2(y)             # shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)          # shape = [32, 64, 1, 1]

        return x * y.expand_as(x)


class SKFusion(nn.Module):
    """Selective Kernel Fusion for multi-scale features"""
    def __init__(self, dim, reduction=16):
        super(SKFusion, self).__init__()
        self.dim = dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim * 3, kernel_size=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        batch_size, C, H, W = x1.shape
        U = x1 + x2 + x3
        Z = self.gap(U)  # (B, C, 1, 1)
        weights = self.fc(Z)  # (B, 3*C, 1, 1)
        weights = weights.view(batch_size, 3, C, 1, 1)  # (B, 3, C, 1, 1)
        weights = self.softmax(weights)

        x_fused = weights[:, 0, :, :, :] * x1 + \
                  weights[:, 1, :, :, :] * x2 + \
                  weights[:, 2, :, :, :] * x3
        return x_fused

class DMRC(nn.Module):
    """
        Deformable Multi-scale Residual Cascade Block
    """
    def __init__(self, dim):
        super(DMRC, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.pre_norm = nn.BatchNorm2d(dim)

        self.conv = nn.Conv2d(dim*2,dim,1,1,0)
        self.conv_in = nn.Conv2d(dim, dim,3,1,1)
        self.conv2 = nn.Conv2d(dim,dim,3,1,1)

        self.conv_d1 = nn.Sequential(
            DeformableConv(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, edge=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.conv_d2 = nn.Sequential(
            DeformableConv(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3, edge=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.conv_d3 = nn.Sequential(
            DeformableConv(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5,edge=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        # self.selayer = SELayer2d(dim)

        self.fusion = SKFusion(dim)
        self.post_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn_post = nn.BatchNorm2d(dim)
        
    
    def forward(self, x, y=None):
        residual = x
        if y is not None:
            y = F.interpolate(y, x.shape[-2:],mode='bilinear',align_corners=False)
            y = self.proj(y)
            x = x - y
            x = self.conv2(x)
            x = x + y
        x = self.pre_norm(x)
        x = self.conv_in(x)
        x1 = self.conv_d1(x)
        x2 = self.conv_d2(x + x1)
        x3 = self.conv_d3(x + x2)

        x_fused = self.fusion(x1,x2,x3)
        x_fused = self.post_conv(x_fused)
        x_fused = self.bn_post(x_fused)
        x_fused = F.gelu(x_fused)
        # x_out = self.deform_conv_out(x_out)
        # x_out = self.selayer(x_fused)
        return residual + x_fused
