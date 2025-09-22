import torch.nn as nn
from torchvision import models

class MobileNetV2Gesture(nn.Module):

    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super(MobileNetV2Gesture, self).__init__()

        try:
            # Try new API first (PyTorch 1.13+)
            if pretrained:
                backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                backbone = models.mobilenet_v2(weights=None)
        except AttributeError:
            # Fall back to old API (PyTorch < 1.13)
            backbone = models.mobilenet_v2(pretrained=pretrained)


        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
