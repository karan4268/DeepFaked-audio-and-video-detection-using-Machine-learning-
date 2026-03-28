# test.py
# Dataset evaluation script

import torch
from torch.utils.data import DataLoader
from Preprocessing.video_preprocessing import FakeAVCeleb3DDataset
from video_model import VideoModel
import multiprocessing
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


def main():

    # ---------------- Device ---------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------- Paths ---------------- #
    root_dir = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"

    checkpoint_path = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\old\3D Cnn Video model old\video_model_best_3D_CNN-Restnet18.pth"

    output_csv = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\old\3D Cnn Video model old\test_predictions_Cross-dataset_fakeAVCelebs.csv"

    # ---------------- Dataset ---------------- #
    test_dataset = FakeAVCeleb3DDataset(
        root_dir=root_dir,
        split="test",
        frames_per_video=24,
        mode="video_binary"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Total test videos: {len(test_dataset)}")

    # ---------------- Load Model ---------------- #
    model = VideoModel().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

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

        for frames, labels, paths in test_loader:

            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)

            probs = torch.softmax(logits, dim=1)[:, 1]

            all_y_true.extend(labels.cpu().numpy())
            all_y_score.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_y_true = np.array(all_y_true)
    all_y_score = np.array(all_y_score)

    print(f"[INFO] Unique labels in test set: {np.unique(all_y_true)}")

    # ---------------- Metrics ---------------- #

    if len(np.unique(all_y_true)) < 2:

        print("[WARNING] Only one class present in test set")

        auc = float("nan")
        best_threshold = 0.5

        optimal_preds = (all_y_score >= 0.5).astype(int)

        optimal_acc = accuracy_score(all_y_true, optimal_preds)

        cm = confusion_matrix(all_y_true, optimal_preds)

    else:

        auc = roc_auc_score(all_y_true, all_y_score)

        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)

        j_scores = tpr - fpr

        best_idx = np.argmax(j_scores)

        best_threshold = thresholds[best_idx]

        optimal_preds = (all_y_score >= best_threshold).astype(int)

        optimal_acc = accuracy_score(all_y_true, optimal_preds)

        cm = confusion_matrix(all_y_true, optimal_preds)

    default_preds = (all_y_score >= 0.5).astype(int)

    default_acc = accuracy_score(all_y_true, default_preds)

    # ---------------- Print Results ---------------- #

    print("\n============= Cross-Dataset Evaluation (FF+ model Test on FakeAVCelebs) (ResNet18 3D-CNN) =============")

    print(f"ROC-AUC                        : {auc:.4f}")
    print(f"Optimal Threshold              : {best_threshold:.4f}")
    print(f"Accuracy @ Optimal Threshold   : {optimal_acc:.4f}")
    print(f"Accuracy @ 0.5 Threshold       : {default_acc:.4f}")

    print("\nConfusion Matrix:")

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
            "pred_class"
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

    print(f"✅ Predictions saved to: {output_csv}")


# ---------------- multiprocesses Entry ---------------- #

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()