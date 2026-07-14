import torch
import torch.nn as nn
from models.alexnet  import CustomAlexNet
from models.vgg16    import CustomVGG16
from models.resnet50 import CustomResNet50


class FusionModel(nn.Module):
    """
    AlexNet(4096) + VGG16(4096) + ResNet50(2048) = 10240 → Grade I/II/III.
    Base model weights loaded from Exp3 checkpoints and frozen during training.
    Only the fusion head is trained.

    Architecture improvements:
      - LayerNorm before the head to normalise heterogeneous backbone features
      - Three-layer MLP head (10240 → 4096 → 2048 → 3) for more capacity
      - Dropout reduced from 0.5 → 0.2 (less aggressive regularisation for a
        single-learnable-layer scenario)
    """

    def __init__(self, num_classes=3, attention_type=None):
        super().__init__()
        # pretrained=False here — weights are loaded from Exp3 checkpoints externally
        self.alexnet = CustomAlexNet( num_classes, activation="GELU",
                                      attention_type=attention_type, pretrained=False)
        self.vgg16   = CustomVGG16(   num_classes, activation="GELU",
                                      attention_type=attention_type, pretrained=False)
        self.resnet  = CustomResNet50(num_classes, activation="GELU",
                                      attention_type=attention_type, pretrained=False)

        # Normalise concatenated features so no single backbone dominates
        self.norm = nn.LayerNorm(10240)

        # Deeper head with moderate dropout
        self.head = nn.Sequential(
            nn.Linear(10240, 4096),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes),
        )

    def freeze_backbones(self):
        for m in (self.alexnet, self.vgg16, self.resnet):
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        f1 = self.alexnet.get_penultimate(x)   # 4096
        f2 = self.vgg16.get_penultimate(x)     # 4096
        f3 = self.resnet.get_penultimate(x)    # 2048
        f  = torch.cat([f1, f2, f3], dim=1)    # 10240
        f  = self.norm(f)
        return self.head(f)