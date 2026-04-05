import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from src.attention import FEM


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # 1x1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # atrous convolutions at different rates
        self.atrous_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1x1(x)
        x2 = self.atrous_6(x)
        x3 = self.atrous_12(x)
        x4 = self.atrous_18(x)

        # global pool branch needs upsampling back to feature size
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size,
                           mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    Baseline DeepLabV3+ with MobileNetV2 backbone.
    No modifications yet — this is the baseline to beat.
    """
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

        # load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # encoder — MobileNetV2 features
        # features[0:4] gives us the low-level features (stride 4)
        # features[0:19] gives us high-level features (stride 16 or 32)
        self.low_level_features = mobilenet.features[:4]   # stride 4
        self.high_level_features = mobilenet.features[4:]  # stride 16+

        # low level feature channels from MobileNetV2 is 24
        # high level feature channels is 1280
        self.low_level_project = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # ASPP on high level features
        self.aspp = ASPP(in_channels=1280, out_channels=256)

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # encoder
        low = self.low_level_features(x)       # stride 4
        high = self.high_level_features(low)   # stride 16+

        # ASPP on high level
        high = self.aspp(high)

        # upsample high level to match low level size
        high = F.interpolate(high, size=low.shape[2:],
                             mode="bilinear", align_corners=False)

        # project low level features
        low = self.low_level_project(low)

        # concatenate and decode
        x = torch.cat([high, low], dim=1)
        x = self.decoder(x)

        # upsample to input size
        x = F.interpolate(x, size=input_size,
                          mode="bilinear", align_corners=False)
        return x

class WeightedFusion(nn.Module):
    """
    Learnable weighted addition of multi-scale shallow features.
    Instead of concatenation, each scale gets a learnable weight.
    w1*shallow1 + w2*shallow2 + w3*shallow3
    """
    def __init__(self, num_scales=3):
        super().__init__()
        # learnable weights, one per scale
        # initialized equally so no scale is favored at start
        self.weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, features):
        # softmax ensures weights sum to 1 and are all positive
        w = torch.softmax(self.weights, dim=0)
        out = sum(w[i] * features[i] for i in range(len(features)))
        return out


class DeepLabV3PlusModified(nn.Module):
    """
    Modified DeepLabV3+ with:
    - FEM attention on shallow features (Change 2 partial)
    - FEM attention on ASPP output (Change 2)
    - Weighted addition of shallow features (Change 1)
    """
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

        # backbone
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # tap three shallow feature levels from MobileNetV2
        # features[0:2]  → stride 4,  channels: 16
        # features[0:4]  → stride 8,  channels: 24
        # features[0:7]  → stride 16, channels: 32
        self.shallow1 = mobilenet.features[:2]   # stride 4
        self.shallow2 = mobilenet.features[2:4]  # stride 8
        self.shallow3 = mobilenet.features[4:7]  # stride 16
        self.high_level = mobilenet.features[7:] # stride 32

        # project all shallow features to same channel dim (48)
        self.proj1 = nn.Sequential(
            nn.Conv2d(16, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(32, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # FEM attention on each shallow feature (Change 2 partial)
        self.fem1 = FEM(in_channels=48, out_channels=48)
        self.fem2 = FEM(in_channels=48, out_channels=48)
        self.fem3 = FEM(in_channels=48, out_channels=48)

        # ASPP on high level features
        self.aspp = ASPP(in_channels=1280, out_channels=256)

        # FEM attention on ASPP output (Change 2)
        self.fem_aspp = FEM(in_channels=256, out_channels=256)

        # weighted fusion of shallow features (Change 1)
        self.weighted_fusion = WeightedFusion(num_scales=3)

        # decoder — takes fused shallow (48) + ASPP (256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # extract shallow features sequentially
        s1 = self.shallow1(x)           # stride 4
        s2 = self.shallow2(s1)          # stride 8
        s3 = self.shallow3(s2)          # stride 16
        high = self.high_level(s3)      # stride 32

        # project to common channel dim
        s1 = self.proj1(s1)
        s2 = self.proj2(s2)
        s3 = self.proj3(s3)

        # FEM attention on each shallow feature
        s1 = self.fem1(s1)
        s2 = self.fem2(s2)
        s3 = self.fem3(s3)

        # upsample s2 and s3 to match s1 size (stride 4)
        s2 = F.interpolate(s2, size=s1.shape[2:],
                           mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=s1.shape[2:],
                           mode="bilinear", align_corners=False)

        # weighted addition of shallow features (Change 1)
        shallow_fused = self.weighted_fusion([s1, s2, s3])

        # ASPP + FEM on high level (Change 2)
        high = self.aspp(high)
        high = self.fem_aspp(high)

        # upsample high to match shallow size
        high = F.interpolate(high, size=shallow_fused.shape[2:],
                             mode="bilinear", align_corners=False)

        # decode
        x = torch.cat([high, shallow_fused], dim=1)
        x = self.decoder(x)

        # upsample to input size
        x = F.interpolate(x, size=input_size,
                          mode="bilinear", align_corners=False)
        return x


def get_modified_model(num_classes=19):
    return DeepLabV3PlusModified(num_classes=num_classes)

def get_model(num_classes=19):
    return DeepLabV3Plus(num_classes=num_classes)