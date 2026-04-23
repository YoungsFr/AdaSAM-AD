import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DeformableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        edge=True,
    ):
        super(DeformableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bias,
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        if edge:
            self.offset_edge = nn.Conv2d(
                256,
                2 * self.kernel_size[0] * self.kernel_size[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )

    def forward(self, x, edge=None):
        offsets = self.offset_conv(x)
        if edge is not None:
            offsets_e = self.offset_edge(edge)
            offsets = offsets * offsets_e

        x = self.deform_conv(x, offsets)
        x = self.bn(x)
        x = self.act(x)
        return x


def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return x