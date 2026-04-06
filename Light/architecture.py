import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Building Block ──────────────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


# ── Encoder ─────────────────────────────────────────────────
class LightEncoder(nn.Module):
    """4 stages, each halves resolution. Outputs 4 feature maps."""
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(           # 1/2
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 32),
        )
        self.stage2 = nn.Sequential(           # 1/4
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64),
        )
        self.stage3 = nn.Sequential(           # 1/8
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128),
        )
        self.stage4 = nn.Sequential(           # 1/16
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 256),
        )

    def forward(self, x):
        c1 = self.stage1(x)   # 16,  H/2
        c2 = self.stage2(c1)  # 64,  H/4
        c3 = self.stage3(c2)  # 128, H/8
        c4 = self.stage4(c3)  # 256, H/16
        return c2, c3, c4


# ── Strip Pooling Module ─────────────────────────────────────
class StripPooling(nn.Module):
    """
    Captures horizontal (roads) and vertical (buildings) context
    cheaply — no heavy ASPP needed.
    """
    def __init__(self, in_ch):
        super().__init__()
        mid = in_ch // 4
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))  # horizontal strip
        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))  # vertical strip
        self.h_conv = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.v_conv = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.fuse   = nn.Sequential(
            nn.Conv2d(in_ch + mid * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        h = F.interpolate(self.h_conv(self.h_pool(x)), size=(H, W), mode='bilinear', align_corners=True)
        v = F.interpolate(self.v_conv(self.v_pool(x)), size=(H, W), mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([x, h, v], dim=1))


# ── Decoder ─────────────────────────────────────────────────
class LightDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # reduce channels before fusion
        self.reduce_c4 = nn.Conv2d(256, 128, 1, bias=False)
        self.reduce_c3 = nn.Conv2d(128,  64, 1, bias=False)
        self.reduce_c2 = nn.Conv2d(64,   32, 1, bias=False)

        # refine after each upsample + skip
        self.refine1 = DepthwiseSeparableConv(128 + 64,  128)
        self.refine2 = DepthwiseSeparableConv(128 + 32,   64)

        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, c2, c3, c4):
        # c4: 1/16 → upsample to 1/8, add c3
        x = F.interpolate(self.reduce_c4(c4), scale_factor=2, mode='bilinear', align_corners=True)
        x = self.refine1(torch.cat([x, self.reduce_c3(c3)], dim=1))

        # 1/8 → upsample to 1/4, add c2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.refine2(torch.cat([x, self.reduce_c2(c2)], dim=1))

        # 1/4 → full resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.head(x)


# ── Full Model ───────────────────────────────────────────────
class FastSegNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.encoder  = LightEncoder()
        self.strip_pool = StripPooling(256)   # applied on deepest features
        self.decoder  = LightDecoder(num_classes)

    def forward(self, x):
        c2, c3, c4 = self.encoder(x)
        c4 = self.strip_pool(c4)              # enrich context before decode
        return self.decoder(c2, c3, c4)


# ── Quick sanity check ───────────────────────────────────────
if __name__ == '__main__':
    model = FastSegNet(num_classes=19)
    x = torch.randn(2, 3, 512, 1024)         # Cityscapes resolution
    out = model(x)
    print("Output shape:", out.shape)         # (2, 19, 512, 1024)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.2f}M")   # ~2.5M