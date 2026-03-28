# =============================================================================
# VideoModel: 3D CNN for Deepfake Video Classification
# Backbone: R3D-18 (pretrained on Kinetics-400)
# Output: Binary classification (real vs fake video)
# =============================================================================
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoModel(nn.Module):
    """
    3D CNN model for deepfake video classification using R3D-18 backbone.

    Default configuration:
        Binary classification (real video vs fake video)
        Output shape: [B, 2]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(VideoModel, self).__init__()

        # ------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)

        # ------------------------------------------------------------
        # Replace classification head
        # ------------------------------------------------------------
        in_features = self.backbone.fc.in_features

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

        Returns:
            logits of shape (B, 2)
        """
        return self.backbone(x)


# =============================================================================
# Model Factory
# =============================================================================
def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "cuda"
):
    """
    Model factory function.
    Ensures consistent class count across training.
    """

    model = VideoModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    return model.to(device)


# =============================================================================
# Sanity Test
# =============================================================================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device=device)

    dummy_input = torch.randn(2, 3, 24, 224, 224).to(device)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expected: [2, 2]