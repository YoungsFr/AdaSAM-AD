import torch
import torch.nn as nn

class MultiScaleAdapter(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.reduced_dim = dim // reduction_ratio

        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(dim))

        self.down_proj = nn.Linear(dim, self.reduced_dim)

        self.dw3x3 = nn.Conv2d(self.reduced_dim, self.reduced_dim, kernel_size=3, padding=1, groups=self.reduced_dim)
        self.dw5x5 = nn.Conv2d(self.reduced_dim, self.reduced_dim, kernel_size=5, padding=2, groups=self.reduced_dim)
        self.dw7x7 = nn.Conv2d(self.reduced_dim, self.reduced_dim, kernel_size=7, padding=3, groups=self.reduced_dim)

        self.conv1x1 = nn.Conv2d(self.reduced_dim, self.reduced_dim, kernel_size=1)

        self.act = nn.GELU()
        self.up_proj = nn.Linear(self.reduced_dim, dim)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut_x = x
        
        x1 = self.norm(x)
        x1 = x1 * self.scale
        x1 = self.down_proj(x1)
        
        x1_spatial = x1.permute(0, 3, 1, 2).contiguous()

        x_dw3 = self.dw3x3(x1_spatial)
        x_dw5 = self.dw5x5(x1_spatial)
        x_dw7 = self.dw7x7(x1_spatial)
        x2 = (x_dw3 + x_dw5 + x_dw7) / 3

        x3 = x1_spatial + x2

        x4 = x3 + self.conv1x1(x3)

        x4_feat = x4.permute(0, 2, 3, 1).contiguous()
        x5 = self.up_proj(self.act(x4_feat))
        
        x6 = shortcut_x + self.gamma * x5
        
        return x6