# =============================================================================
# test_audio.py
# Testing script for FakeAVCeleb Audio Model (uses same dataset + cache)
# =============================================================================

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from data_loader import AudioDataset, get_cache_dir   

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def evaluate(model, loader, device):

    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            logits = model(x).squeeze(1)

            # ✅ MUST MATCH TRAIN
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = roc_auc_score(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    return auc, eer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------------------------------
    # USE SAME PATHS AS TRAINING
    # -------------------------------------------------------------------------
    DATASET = AudioDataset(
        split="test"   # or "val" depending on your design
    )

    loader = DataLoader(
        DATASET,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"[INFO] Samples: {len(DATASET)}")

    model = AudioResNet18().to(device)

    MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\audio_model_best.pth"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("[INFO] Model loaded")

    auc, eer = evaluate(model, loader, device)

    print("\n==============================")
    print(f"[EVAL] AUC : {auc:.6f}")
    print(f"[EVAL] EER : {eer:.6f}")
    print("==============================\n")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()