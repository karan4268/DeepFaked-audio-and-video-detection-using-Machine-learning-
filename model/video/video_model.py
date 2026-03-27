import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoModel(nn.Module):
    """
    3D CNN model for deepfake video classification using R3D-18 backbone.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super(VideoModel, self).__init__()

        # Load pretrained R3D-18 backbone
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)

        # Get input features of final fully connected layer
        in_features = self.backbone.fc.in_features

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expected input shape:
            (B, C, T, H, W)
        Example:
            (2, 3, 24, 224, 224)
        """
        return self.backbone(x)


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "cuda"
):
    """
    Model factory function.
    """
    model = VideoModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    model = model.to(device)

    return model


if __name__ == "__main__":
    # Quick sanity test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device=device)

    dummy_input = torch.randn(2, 3, 24, 224, 224).to(device)
    output = model(dummy_input)

    print("Output shape:", output.shape)
