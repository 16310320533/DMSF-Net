# coding=utf8
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from modules import HybridConvAttention, MultiScaleDynamicFusion


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class DMSFNet(nn.Module):
    """
    DMSF-Net: Synergistic Integration of Deformable Convolution
    and Attention Gating.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained)

        # Contribution 1: HCA Modules integrated into encoder stages
        self.trans_blocks = nn.ModuleList([
            HybridConvAttention(512),
            HybridConvAttention(1024),
            HybridConvAttention(2048)
        ])

        # Contribution 3: Hierarchical Skip Connections (Channel Adaptation)
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(2048, 256, 1)
        ])

        # Decoder Construction
        self.fusion = MultiScaleDynamicFusion(256)

        self.decoder_stages = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.GELU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, num_classes, 1)
            )
        ])

    def forward(self, x):
        # Encoding Path
        features = self.encoder(x)

        # Apply HCA to high-level features
        trans_feats = [block(feat) for block, feat in zip(self.trans_blocks, features[1:])]

        # Process Hierarchical Skip Connections
        skips = [
            self.skip_convs[0](features[0]),
            self.skip_convs[1](trans_feats[0]),
            self.skip_convs[2](trans_feats[1]),
            self.skip_convs[3](trans_feats[2])
        ]

        # Multi-scale Dynamic Fusion
        x = self.fusion(skips)

        # Progressive Decoding
        x = self.decoder_stages[0](x)
        x = self.decoder_stages[1](x)
        x = self.decoder_stages[2](x)

        return torch.sigmoid(x)