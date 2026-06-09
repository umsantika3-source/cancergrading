import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from models.attention import build_attention


class CustomResNet50(nn.Module):
    """
    ResNet50 backbone with optional attention after each stage (layer1-4).
    When activation="GELU", a GELU is applied after each stage output.
    get_penultimate() returns 2048-dim features for FusionModel.
    """

    def __init__(self, num_classes=3, activation="ReLU", attention_type=None, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1   # 256 ch
        self.layer2  = base.layer2   # 512 ch
        self.layer3  = base.layer3   # 1024 ch
        self.layer4  = base.layer4   # 2048 ch
        self.avgpool = base.avgpool  # → (B, 2048, 1, 1)

        # Attention after each stage
        self.attn1 = build_attention(attention_type, 256)
        self.attn2 = build_attention(attention_type, 512)
        self.attn3 = build_attention(attention_type, 1024)
        self.attn4 = build_attention(attention_type, 2048)

        self.use_gelu = (activation == "GELU")
        self.head = nn.Linear(2048, num_classes)

    def get_penultimate(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.attn1 is not None: x = self.attn1(x)
        if self.use_gelu: x = F.gelu(x)

        x = self.layer2(x)
        if self.attn2 is not None: x = self.attn2(x)
        if self.use_gelu: x = F.gelu(x)

        x = self.layer3(x)
        if self.attn3 is not None: x = self.attn3(x)
        if self.use_gelu: x = F.gelu(x)

        x = self.layer4(x)
        if self.attn4 is not None: x = self.attn4(x)
        if self.use_gelu: x = F.gelu(x)

        return self.avgpool(x).flatten(1)   # (B, 2048)

    def forward(self, x):
        return self.head(self.get_penultimate(x))
