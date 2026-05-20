# video_model.py
# =============================================================================
# R3D-18 Video Model for Deepfake Detection
#
# CHANGES FROM ORIGINAL:
#   [FUSION] Added return_embedding flag — when True, returns 512-dim backbone
#            features before the FC head instead of class logits.
#            Mirrors AudioResNet18.return_embedding for fusion layer alignment.
#
#   [UNCHANGED] All training behaviour identical to original.
# =============================================================================

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoModel(nn.Module):
    """
    3D CNN model for deepfake video classification using R3D-18 backbone.

    Args:
        num_classes (int)     : number of output classes (default 2)
        pretrained  (bool)    : load Kinetics-400 pretrained weights
        dropout     (float)   : dropout before final FC
        return_embedding (bool): if True, forward() returns 512-dim features
                                 instead of logits — used by fusion layer

    Input shape  : (B, C, T, H, W)  e.g. (2, 3, 24, 224, 224)
    Output shape :
        return_embedding=False → (B, num_classes)  logits
        return_embedding=True  → (B, 512)          backbone features
    """

    def __init__(
        self,
        num_classes      : int  = 2,
        pretrained       : bool = True,
        dropout          : float = 0.3,
        return_embedding : bool = False
    ):
        super(VideoModel, self).__init__()

        self.return_embedding = return_embedding

        # Load backbone
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)

        # Feature size before FC
        in_features = self.backbone.fc.in_features   # 512

        # Replace FC with dropout + classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        For return_embedding=True we extract features before the FC head
        by temporarily bypassing it. This avoids needing a separate model
        and keeps checkpoint compatibility — same weights work for both
        classification and embedding extraction.
        """
        if self.return_embedding:
            # Run all layers except final FC
            # R3D-18 structure: stem → layer1-4 → avgpool → flatten → fc
            x = self.backbone.stem(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = x.flatten(1)          # (B, 512)
            return x
        else:
            return self.backbone(x)   # (B, num_classes)


def build_model(
    num_classes      : int   = 2,
    pretrained       : bool  = True,
    dropout          : float = 0.3,
    return_embedding : bool  = False,
    device           : str   = "cuda"
) -> VideoModel:
    """Model factory."""
    model = VideoModel(
        num_classes      = num_classes,
        pretrained       = pretrained,
        dropout          = dropout,
        return_embedding = return_embedding
    )
    return model.to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test classification mode
    model_cls = build_model(device=device, return_embedding=False)
    x = torch.randn(2, 3, 24, 224, 224).to(device)
    out = model_cls(x)
    print(f"Classification output shape : {out.shape}")   # (2, 2)

    # Test embedding mode
    model_emb = build_model(device=device, return_embedding=True)
    # Load same weights — embeddings work with existing checkpoints
    model_emb.load_state_dict(model_cls.state_dict())
    emb = model_emb(x)
    print(f"Embedding output shape      : {emb.shape}")   # (2, 512)