import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Channel recalibration: squeeze → excitation → scale."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel attention + spatial attention in sequence."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = _ChannelAttention(channels, reduction)
        self.spatial_att = _SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class MaskGuidedAttention(nn.Module):
    """
    Uses binary segmentation mask as a spatial prior.
    Phase 2 only — requires running HoVer-Net / U-Net first to produce masks.
    """

    def forward(self, features, mask):
        mask_resized = F.interpolate(mask, features.shape[2:])
        return features * mask_resized + features  # residual so gradient flows


def build_attention(attention_type, channels):
    """Factory — returns an attention module or None."""
    if attention_type is None or attention_type == "None":
        return None
    if attention_type == "SE":
        return SEBlock(channels)
    if attention_type == "CBAM":
        return CBAM(channels)
    if attention_type in ("HoVerNet", "MaskMitosis", "DKSUNet", "UNet"):
        return MaskGuidedAttention()
    raise ValueError(f"Unknown ATTENTION_TYPE: {attention_type!r}")
