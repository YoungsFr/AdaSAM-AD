import torch
import torch.nn.functional as F
import torch.nn as nn

from .encoder import ImageEncoder
from .decoder import SegHead, HPPF
from .DSFE import DSFE
from .DMRC import DMRC



class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class FreqPromptGenerator(nn.Module):
    
    def __init__(self, in_channels=72, hidden_channels=72, out_channels=72):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(hidden_channels)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(hidden_channels)
        self.gelu2 = nn.GELU()
        self.CBAM = CBAM(in_channels,9,7)
        
        self.deconv = nn.ConvTranspose2d(hidden_channels, out_channels, 
                                          kernel_size=4, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(out_channels)
        self.gelu3 = nn.GELU()
    
    def forward(self, x):
        
        fft_result = torch.fft.fft2(x) 
        phase = torch.angle(fft_result)
        
        
        out = self.conv1(phase)
        out = self.bn1(out)
        out = self.gelu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        
        out = self.CBAM(out)


        out = self.deconv(out)   # [B, out_channels, H*2, W*2]
        out = self.bn3(out)
        out = self.gelu3(out)
        return out

class AdaSAM(nn.Module):
    def __init__(self, sam2_model):
        super(AdaSAM, self).__init__()
        
        """ Modules for Ada-SAM """
        self.sam_encoder = ImageEncoder(sam2_model)

        self.down_sample = nn.Sequential(
            nn.Conv2d(3,72,kernel_size=7,padding=3,stride=2),
            nn.BatchNorm2d(72),
            nn.GELU()
        )

        self.freq_prompt_gen = FreqPromptGenerator(in_channels=72, out_channels=72)
        self.sigmoid = nn.Sigmoid()
        
        self.DSFE1 = DSFE(72)
        self.DSFE2 = DSFE(144)
        self.DSFE3 = DSFE(288)
        self.DSFE4 = DSFE(576, use_attention=False)
        self.DSFE5 = DSFE(1152, use_attention=False)

        self.DMRC1 = DMRC(72)
        self.DMRC2 = DMRC(144)
        self.DMRC3 = DMRC(288)
        self.DMRC4 = DMRC(576)
        self.DMRC5 = DMRC(1152)
        
        self.seg_head5 = SegHead(1152)
        self.seg_head4 = SegHead(576)
        self.seg_head3 = SegHead(288)
        self.seg_head2 = SegHead(144)
        self.HPPF = HPPF(72, 112,prompt_channels=72)

    def forward(self, x):
        f1 = self.down_sample(x)
        f2, f3, f4, f5 = self.sam_encoder(x)
        freq_prompt = self.freq_prompt_gen(f1)

        f5 = self.DMRC5(f5, None)
        mask5 = self.seg_head5(f5)

        f4 = self.DSFE4(f4, mask5)
        f4 = self.DMRC4(f4, f5)
        mask4 = self.seg_head4(f4)

        f3 = self.DSFE3(f3,mask4)
        f3 = self.DMRC3(f3, f4)
        mask3 = self.seg_head3(f3)

        f2 = self.DSFE2(f2,mask3)
        f2 = self.DMRC2(f2, f3)
        mask2 = self.seg_head2(f2)

        f1 = self.DSFE1(f1,mask2)
        f1 = self.DMRC1(f1, f2)

        mask1 = self.HPPF(f1,f2,f3,freq_prompt)

        return [mask1, mask2, mask3, mask4, mask5]


