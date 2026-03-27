# =============================================================================
# test_audio.py
# Final Evaluation Script for Deepfake Audio Detection (ASVspoof 2019 LA - TEST)
# Uses index-based cache pipeline (aligned with training)
# =============================================================================

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from data_loader import ASVSpoofDataset


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x).squeeze(1)

            # Convert logits → probabilities
            probs = torch.sigmoid(logits)

            all_scores.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Merge batches
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    # Safety check
    if len(np.unique(all_labels)) < 2:
        raise ValueError("Only one class present. Cannot compute AUC/EER.")

    auc = roc_auc_score(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    return auc, eer, all_scores, all_labels


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    CACHE_BASE = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache"

    MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\audio_model_best.pth"

    OUTPUT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\eval_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATASET (INDEX-BASED)
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    eval_dataset = ASVSpoofDataset(
        CACHE_BASE,
        split="test"   # IMPORTANT: must match cache split
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    print(f"[INFO] Eval samples: {len(eval_dataset)}")

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Best model not found.")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("[INFO] Loaded best checkpoint")

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    auc, eer, scores, labels = evaluate(model, eval_loader, device)

    print("\n==============================")
    print(f"[TEST] AUC : {auc:.6f}")
    print(f"[TEST] EER : {eer:.6f}")
    print("==============================\n")

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS (FOR FUSION)
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "audio_test_scores.npy"), scores)
    np.save(os.path.join(OUTPUT_DIR, "audio_test_labels.npy"), labels)

    print(f"[INFO] Saved outputs to: {OUTPUT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()