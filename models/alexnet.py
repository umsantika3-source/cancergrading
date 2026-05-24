import torch.nn as nn
from torchvision import models
from models.attention import build_attention


class CustomAlexNet(nn.Module):
    """
    AlexNet backbone with configurable activation (ReLU/GELU) and optional attention.
    Attention is inserted after features block, before avgpool.
    get_penultimate() returns 4096-dim features for FusionModel.
    """

    def __init__(self, num_classes=3, activation="ReLU", attention_type=None, pretrained=True):
        super().__init__()
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.alexnet(weights=weights)

        self.features  = base.features   # outputs (B, 256, H', W')
        self.avgpool   = base.avgpool    # AdaptiveAvgPool2d → (B, 256, 6, 6)
        self.attention = build_attention(attention_type, 256)

        act = nn.GELU() if activation == "GELU" else nn.ReLU(inplace=True)
        act2 = nn.GELU() if activation == "GELU" else nn.ReLU(inplace=True)

        self.penultimate = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            act,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            act2,
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
