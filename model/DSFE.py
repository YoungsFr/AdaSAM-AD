import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .DC import DeformableConv

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class LayerNorm(nn.LayerNorm):
    def __init__(self, inchannels):
        super().__init__(inchannels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

class SelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        ffn_expansion_factor: int = 4,
    ):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim**-0.5

        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(
            channels, ffn_expansion_factor=ffn_expansion_factor, bias=False
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        x_norm1 = self.norm1(x)
        if mask is not None:
            mask = F.interpolate(
                mask, size=(h, w), mode="bilinear", align_corners=False
            )
        q = self.conv1(x_norm1)
        k = self.conv2(x_norm1)
        v = self.conv3(x_norm1)
        k = k * mask if mask is not None else k
        v = v * mask if mask is not None else v
        q = rearrange(q, "b (head d) h w -> b head d (h w)", head=self.num_heads)
        k = rearrange(k, "b (head d) h w -> b head d (h w)", head=self.num_heads)
        v = rearrange(v, "b (head d) h w -> b head d (h w)", head=self.num_heads)

        dots = (q@ k.transpose(-2, -1)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b head d (h w) -> b (head d) h w", head=self.num_heads, h=h, w=w)

        attn_out = self.proj_dropout(self.proj(out))
        x = x + attn_out
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))

        channel_weights = self.sigmoid(avg_out + max_out)

        return channel_weights.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return spatial_weights


class DSFE(nn.Module):
    """Dual-Stream Feature Enhancement"""
    def __init__(self, dim, reduction_ratio=16, kernel_size=7, num_heads=8, use_attention=True):

        super(DSFE, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.DC = DeformableConv(dim,dim,3,1,1,edge=False)
        else:
            self.DC = nn.Conv2d(dim,dim,3,1,1)
        self.BN = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        self.channel_attention = ChannelAttention(dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        mask = F.interpolate(mask,x.shape[-2:],mode='bilinear',align_corners=False)
        mask = 1 - self.sigmoid(mask)
        residual = x
        x = x * mask
        x = self.DC(x)
        x = self.BN(x)
        x = self.activation(x)
        x = self.channel_attention(x) * x
        if self.use_attention:
            x = self.spatial_attention(x) * x
        x = x + residual
        return x


if __name__ == '__main__':
    a = torch.randn([5,256,7,7])
    b = DSFE(256)
    c, mask = b(a)
    print(mask.shape)