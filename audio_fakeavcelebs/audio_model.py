# model.py
# Improved Audio ResNet18 for Binary Deepfake Detection

import torch
import torch.nn as nn
import torchvision.models as models


class AudioResNet18(nn.Module):
    def __init__(self, return_embedding=False):
        super().__init__()

        self.return_embedding = return_embedding

        # ✅ Use pretrained weights (IMPORTANT)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ✅ Modify first conv for 1-channel
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove FC
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ✅ Better classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x: (B, 1, 128, T)
        """

        features = self.backbone(x)  # (B, 512)

        if self.return_embedding:
            return features

        logits = self.classifier(features)  # (B, 1)

        return logits