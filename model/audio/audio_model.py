# audio_model.py
# =============================================================================
# Improved Audio ResNet18 (Deepfake Detection Optimized)
# =============================================================================
#
# CHANGES FROM ORIGINAL:
#
#   [FIX 1] BatchNorm1d(256) replaced with LayerNorm(256) in classifier head.
#           BatchNorm1d computes statistics across the batch dimension. When
#           the weighted sampler pulls similar clips into the same batch, all
#           512-dim backbone features can be near-identical → batch variance
#           ≈ 0 → (x - mean) / sqrt(var + ε) → NaN. This is intermittent,
#           which is exactly the pattern seen in the logs (some batches fine,
#           some NaN). LayerNorm normalizes across the feature dimension per
#           sample, so it is immune to batch composition — variance is always
#           healthy because it's computed over 256 features, not B samples.
#
#   [FIX 2] Pretrained conv1 weights are averaged into the new 1-channel conv
#           instead of being randomly re-initialized. The original code creates
#           a fresh random conv1, which means the backbone receives activations
#           from an untrained layer while all subsequent layers expect ImageNet-
#           style features. This causes large unstable activations early in
#           training. Averaging the 3 pretrained input channels into 1 is the
#           standard practice (used in medical imaging, audio, etc.) and gives
#           the backbone a stable warm start.
# =============================================================================

import torch
import torch.nn as nn
import torchvision.models as models


class AudioResNet18(nn.Module):
    def __init__(self, pretrained=True, return_embedding=False):
        super(AudioResNet18, self).__init__()

        self.return_embedding = return_embedding

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ----------------------------------------------------
        # [FIX 2] Replace conv1 with 1-channel input, preserving
        # pretrained weights by averaging across the 3 colour channels.
        # Random re-init (original) gives the backbone garbage activations
        # from the very first layer, causing unstable training early on.
        # ----------------------------------------------------
        old_conv1   = self.backbone.conv1          # shape: [64, 3, 7, 7]
        new_conv1   = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Average the 3 RGB channel weights into a single channel.
            # This preserves the learned spatial filters.
            with torch.no_grad():
                new_conv1.weight.copy_(
                    old_conv1.weight.mean(dim=1, keepdim=True)
                )
        # else: random init is fine — nothing pretrained to transfer

        self.backbone.conv1 = new_conv1

        # ----------------------------------------------------
        # Feature extractor (strip original FC)
        # ----------------------------------------------------
        in_features         = self.backbone.fc.in_features   # 512
        self.backbone.fc    = nn.Identity()

        # ----------------------------------------------------
        # Classification head
        # [FIX 1] BatchNorm1d → LayerNorm to prevent NaN from
        # near-zero batch variance when similar clips are batched.
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),              # [FIX 1] was BatchNorm1d(256)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(64, 1)                # binary logit
        )

    def forward(self, x):
        """
        x : (B, 1, 128, T)
        returns logits : (B, 1)
        """
        features = self.backbone(x)         # (B, 512)

        if self.return_embedding:
            return features

        logits = self.classifier(features)  # (B, 1)
        return logits