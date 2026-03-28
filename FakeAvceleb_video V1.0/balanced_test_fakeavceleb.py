# =============================================================================
# test_balanced_json.py
# Evaluate video model using a custom JSON (balanced RR vs FR)
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import multiprocessing
import csv

from video_model import VideoModel

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# 🔷 Custom Dataset (JSON-based)
# =============================================================================

class JSONVideoDataset(Dataset):
    def __init__(self, json_path, frames_per_video=24):

        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = data["test"]
        self.frames_per_video = frames_per_video

        print(f"[INFO] Loaded {len(self.samples)} samples from JSON")

    def __len__(self):
        return len(self.samples)

    def load_frames(self, path):
        """
        Assumes your cached_faces directory contains frames as images
        """
        import cv2

        frame_files = sorted(os.listdir(path))

        # Sample frames uniformly
        if len(frame_files) >= self.frames_per_video:
            indices = np.linspace(0, len(frame_files) - 1, self.frames_per_video).astype(int)
            selected = [frame_files[i] for i in indices]
        else:
            selected = frame_files

        frames = []

        for fname in selected:
            fpath = os.path.join(path, fname)
            img = cv2.imread(fpath)

            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW

            frames.append(img)

        # Pad if needed
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        frames = np.stack(frames)              # [T, C, H, W]
        frames = np.transpose(frames, (1, 0, 2, 3))  # → [C, T, H, W]

        return torch.tensor(frames, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        path = sample["path"]
        label = sample["primary_label"]

        # 🔥 Video binary mapping (RR vs FR only)
        label = 1 if label == 1 else 0

        frames = self.load_frames(path)

        return frames, label, path


# =============================================================================
# 🔷 Main Evaluation
# =============================================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------- Paths ---------------- #
    json_path = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\fakeav_balanced_rr_fr.json"

    checkpoint_path = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\old\3D Cnn Video model old\video_model_best_3D_CNN-Restnet18.pth"

    output_csv = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\test_balanced_predictions_fakeavceleb.csv"

    # ---------------- Dataset ---------------- #
    dataset = JSONVideoDataset(json_path=json_path, frames_per_video=24)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------------- Load Model ---------------- #
    model = VideoModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[INFO] Loaded model")

    # ---------------- Evaluation ---------------- #
    all_y_true = []
    all_y_score = []
    all_paths = []

    with torch.no_grad():

        for frames, labels, paths in loader:

            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_y_true.extend(labels.cpu().numpy())
            all_y_score.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_y_true = np.array(all_y_true)
    all_y_score = np.array(all_y_score)

    print(f"[INFO] Unique labels: {np.unique(all_y_true)}")

    # ---------------- Metrics ---------------- #
    auc = roc_auc_score(all_y_true, all_y_score)

    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    optimal_preds = (all_y_score >= best_threshold).astype(int)
    optimal_acc = accuracy_score(all_y_true, optimal_preds)

    default_preds = (all_y_score >= 0.5).astype(int)
    default_acc = accuracy_score(all_y_true, default_preds)

    cm = confusion_matrix(all_y_true, optimal_preds)

    # ---------------- Print ---------------- #
    print("\n============= Balanced RR vs FR Evaluation =============")
    print(f"ROC-AUC                        : {auc:.4f}")
    print(f"Optimal Threshold              : {best_threshold:.4f}")
    print(f"Accuracy @ Optimal Threshold   : {optimal_acc:.4f}")
    print(f"Accuracy @ 0.5 Threshold       : {default_acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_y_true, optimal_preds, digits=4))

    print("========================================================\n")

    # ---------------- Save CSV ---------------- #
    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["path", "true", "prob_fake", "pred"])

        for p, y, s, pred in zip(all_paths, all_y_true, all_y_score, optimal_preds):
            writer.writerow([p, int(y), float(s), int(pred)])

    print(f"✅ Saved predictions to: {output_csv}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()