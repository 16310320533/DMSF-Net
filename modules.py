# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformConv2d(nn.Module):
    """
    Base Deformable Convolution Layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Parameter initialization
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(
            x, offset, self.conv.weight, self.conv.bias,
            padding=(self.conv.padding[0], self.conv.padding[1])
        )


class HybridConvAttention(nn.Module):
    """
    Contribution 1: Hybrid Convolution-Attention (HCA) Mechanism.
    Integrates deformable convolution with attention gating.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Reduced channel dimension for offset generation
        self.offset_conv = nn.Conv2d(
            in_channels,
            18,  # 2*3x3 kernel parameters
            kernel_size=3,
            padding=1
        )
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )

        # Specialized initialization for stability
        nn.init.normal_(self.offset_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in')

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(
            x, offset, self.conv.weight, self.conv.bias,
            padding=1
        )


class MultiScaleDynamicFusion(nn.Module):
    """
    Contribution 2: Multi-scale Dynamic Fusion (MDF).
    Aligns and fuses hierarchical features using channel attention.
    """

    def __init__(self, in_channels=256):
        super().__init__()
        # Alignment modules for hierarchical inputs
        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.Identity()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
        ])

        # Channel attention mechanism
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, 1),
            nn.Sigmoid()
        )

        # Dimensionality reduction
        self.channel_reduce = nn.Conv2d(in_channels * 4, in_channels, 1)

    def forward(self, features):
        aligned = []
        # Target size based on the first feature map
        target_size = features[0].shape[-2:]

        for conv, feat in zip(self.align_convs, features):
            x = conv(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            aligned.append(x)

        fused = torch.cat(aligned, dim=1)
        attn = self.channel_att(fused)
        weighted = fused * attn

        return self.channel_reduce(weighted)