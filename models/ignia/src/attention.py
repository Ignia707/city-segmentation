import torch
import torch.nn as nn


class FEM(nn.Module):
    """
    Feature Enhancement Module (FEM).
    Dual-branch architecture combining standard and dilated convolutions
    to capture multi-scale features and enhance small object detection.

    Architecture:
        Branch 1: 1x1 conv → 3x3 conv
        Branch 2: 1x1 conv → 1x3 conv → 3x1 conv → 3x3 dilated conv
        Branch 3: 1x1 conv → 3x1 conv → 1x3 conv → 3x3 dilated conv
        Branch 4: 1x1 conv (identity-like)
        Output: concat(B1, B2, B3, B4) + 1x1 conv(input)
    """
    def __init__(self, in_channels, out_channels=None, dilation=5):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # project input to out_channels for residual
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # branch 1: standard 3x3 conv path
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # branch 2: asymmetric + dilated conv path
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # branch 3: asymmetric (flipped order) + dilated conv path
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # branch 4: simple 1x1 conv
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # fuse all branches
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # concatenate all branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)

        # residual addition with projected input
        return out + self.input_proj(x)