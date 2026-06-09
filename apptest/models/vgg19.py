import torch.nn as nn
from torchvision import models
from models.attention import build_attention


class CustomVGG19(nn.Module):
    """
    VGG19 backbone with configurable activation (ReLU/GELU) and optional attention.
    Attention is inserted after features block, before avgpool.
    get_penultimate() returns 4096-dim features for FusionModel.
    
    Key difference from VGG16: VGG19 has 3 additional convolutional layers
    (one more block of conv layers), but same 4096-dim penultimate layer.
    """

    def __init__(self, num_classes=3, activation="ReLU", attention_type=None, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.vgg19(weights=weights)

        self.features  = base.features   # outputs (B, 512, H', W')
        self.avgpool   = base.avgpool    # AdaptiveAvgPool2d → (B, 512, 7, 7)
        self.attention = build_attention(attention_type, 512)

        act  = nn.GELU() if activation == "GELU" else nn.ReLU(inplace=True)
        act2 = nn.GELU() if activation == "GELU" else nn.ReLU(inplace=True)

        # VGG19 flatten: same as VGG16 - 512 * 7 * 7 = 25088
        # (The extra conv layers don't change spatial dimensions before avgpool)
        self.penultimate = nn.Sequential(
            nn.Linear(25088, 4096),
            act,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            act2,
            nn.Dropout(),
        )
        self.head = nn.Linear(4096, num_classes)

    def get_penultimate(self, x):
        x = self.features(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.avgpool(x)
        return self.penultimate(x.flatten(1))

    def forward(self, x):
        return self.head(self.get_penultimate(x))