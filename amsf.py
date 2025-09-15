import torch.nn as nn
import torch.nn.functional as F

class AMSF(nn.Module):
    """自适应多尺度融合模块"""
    def __init__(self, in_channels_list, out_channels):
        super(AMSF, self).__init__()
        self.fusion_conv = nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1)

    def forward(self, features):
        upsampled = [
            F.interpolate(f, size=features[0].shape[2:], mode='bilinear', align_corners=True)
            for f in features[1:]
        ]
        return self.fusion_conv(nn.functional.relu(torch.cat([features[0]] + upsampled, dim=1)))
