import torch
import torch.nn as nn
import torch.nn.functional as F
from .mona import MultiScaleAdapter

class ImageEncoder(nn.Module):
    def __init__(self, sam2_model):
        super(ImageEncoder, self).__init__()


        self.scalp = sam2_model.image_encoder.scalp
        self.hiera = sam2_model.image_encoder.trunk
        self.blocks = self.init_blocks()


    def init_blocks(self):
        blocks = nn.ModuleList()
        channel_list = [144, 288, 576, 1152]
        for i in range(0, len(channel_list)):
            channel = channel_list[i]
            block = MultiScaleAdapter(channel)
            blocks.append(block)
        return blocks


    def forward(self, x):
        x = self.hiera.patch_embed(x)
        x = x + self.hiera._get_pos_embed(x.shape[1:3])  # B, H/4, W/4, C

        hiera_output = []
        j = 0
        k = 0
        for i, blk in enumerate(self.hiera.blocks):
            if (i == self.hiera.stage_ends[-1]) or (i in self.hiera.stage_ends and self.hiera.return_interm_layers):
                x = blk(x)  # B,H,W,C
                residual = x
                x = self.blocks[j](x)
                x = x + residual
                out_x  = x.permute(0, 3, 1, 2)  # B,C,H,W
                hiera_output.append(out_x)
                j = j + 1
            else:
                x = blk(x)
        return hiera_output
