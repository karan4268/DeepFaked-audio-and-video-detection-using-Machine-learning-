# test.py
# test.py # Cross-dataset evaluation script 
# (NOTE:- Check cache flags in pereprocessing and data loader frist # - cache faces are camputed for FF++ and can be used for Training, eval and Testing on F++. 
# - **For Celeb-DF, cache faces are not computed Becasue Celeb-DB was used for Cross Dataset Evaluation (Unseen Videos) of trained video model, so set use_cache = False in video_preprocessing.py and data_loader.py When using Celeb-DB **)

import torch
from torch.utils.data import DataLoader
from Preprocessing.video_preprocessing import DeepfakedDataset
from video_model import VideoModel
import os
import csv
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------- Paths ---------------- #
root_dir = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\celeb DB"
checkpoint_path = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\old\3D Cnn Video model old\video_model_best_3D_CNN-Restnet18.pth"
output_csv = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\old\3D Cnn Video model old\test_predictions_Cross-dataset.csv"

# ---------------- Dataset ---------------- #
test_dataset = DeepfakedDataset(
    root_dir=root_dir,
    split="test",
    frames_per_video=24,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
)

print(f"[INFO] Total test videos: {len(test_dataset)}")

# ---------------- Load Model ---------------- #
model = VideoModel().to(device)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"[ERROR] Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle both checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"[INFO] Loaded model from: {checkpoint_path}")

# ---------------- Evaluation ---------------- #
all_y_true = []
all_y_score = []
all_paths = []

with torch.no_grad():
    for frames, labels, in test_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of fake class

        all_y_true.extend(labels.cpu().numpy())
        all_y_score.extend(probs.cpu().numpy())

all_y_true = np.array(all_y_true)
all_y_score = np.array(all_y_score)

print(f"[INFO] Unique labels in test set: {np.unique(all_y_true)}")

# ---------------- Metrics ---------------- #

if len(np.unique(all_y_true)) < 2:
    print("[WARNING] Only one class present in test set. ROC-AUC cannot be computed.")
    auc = float("nan")
    best_threshold = 0.5
    optimal_preds = (all_y_score >= 0.5).astype(int)
    optimal_acc = accuracy_score(all_y_true, optimal_preds)
    cm = confusion_matrix(all_y_true, optimal_preds)
else:
    # ROC-AUC
    auc = roc_auc_score(all_y_true, all_y_score)

    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Predictions using optimal threshold
    optimal_preds = (all_y_score >= best_threshold).astype(int)
    optimal_acc = accuracy_score(all_y_true, optimal_preds)

    cm = confusion_matrix(all_y_true, optimal_preds)

# Accuracy at default 0.5 threshold (for calibration comparison)
default_preds = (all_y_score >= 0.5).astype(int)
default_acc = accuracy_score(all_y_true, default_preds)

# ---------------- Print Results ---------------- #

print("\n============= Cross-Dataset Evaluation (ResNet18 3D-CNN) =============")
print(f"ROC-AUC                        : {auc:.4f}")
print(f"Optimal Threshold              : {best_threshold:.4f}")
print(f"Accuracy @ Optimal Threshold   : {optimal_acc:.4f}")
print(f"Accuracy @ 0.5 Threshold       : {default_acc:.4f}")
print("\nConfusion Matrix (Optimal Threshold):")
print(cm)
print("\nClassification Report:")
print(classification_report(all_y_true, optimal_preds, digits=4))
print("================================================================\n")

# ---------------- Save Predictions ---------------- #
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video_path",
        "true_label",
        "pred_prob_fake",
        "pred_class_optimal_threshold"
    ])

    for path, y_true, y_score, y_pred in zip(
        all_paths, all_y_true, all_y_score, optimal_preds
    ):
        writer.writerow([
            path,
            int(y_true),
            round(float(y_score), 6),
            int(y_pred)
        ])

print(f"✅ Per-video predictions saved to: {output_csv}")