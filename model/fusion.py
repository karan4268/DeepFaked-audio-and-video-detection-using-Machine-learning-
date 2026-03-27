import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, video_feat, audio_feat):
        fused = torch.cat((video_feat, audio_feat), dim=1)
        return self.classifier(fused)
