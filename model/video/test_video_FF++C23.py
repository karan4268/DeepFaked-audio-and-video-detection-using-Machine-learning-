# =============================================================================
# test_video_FF++ C23.py
# Cross-dataset evaluation for Video Model (FF++)
#
# Outputs:
#   - Overall AUC + Accuracy
#   - Optimal threshold (Youden’s J)
#   - Confusion Matrix + Classification Report
#   - (Optional) Per-group metrics (e.g., per dataset / attack)
#   - video_scores.npy + video_labels.npy  (for fusion)
#
# =============================================================================

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from collections import defaultdict

from Preprocessing.video_preprocessing import DeepfakedDataset
from video_model import VideoModel


# =============================================================================
# DEFAULT PATHS (EDIT THESE)
# =============================================================================

DEFAULT_ROOT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23" 
)

DEFAULT_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video_combined\combined_train_best.pth"
)

DEFAULT_OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video\eval_outputs\FF++C23"
)


# =============================================================================
# METRICS
# =============================================================================

def compute_optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return thresholds[idx]


def print_main_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2:
        print("[WARNING] Only one class present. Skipping ROC-AUC.")
        return None, 0.5

    auc = roc_auc_score(labels, scores)
    best_thr = compute_optimal_threshold(labels, scores)

    preds_opt = (scores >= best_thr).astype(int)
    preds_def = (scores >= 0.5).astype(int)

    acc_opt = accuracy_score(labels, preds_opt)
    acc_def = accuracy_score(labels, preds_def)

    print("\n========== Overall Metrics ==========")
    print(f"AUC                     : {auc:.4f}")
    print(f"Optimal Threshold       : {best_thr:.4f}")
    print(f"Accuracy @ Optimal Thr  : {acc_opt:.4f}")
    print(f"Accuracy @ 0.5 Thr      : {acc_def:.4f}")

    print("\nConfusion Matrix (Optimal):")
    print(confusion_matrix(labels, preds_opt))

    print("\nClassification Report:")
    print(classification_report(labels, preds_opt, digits=4))

    return auc, best_thr


def print_group_metrics(group_dict):
    print("\n========== Per-Group Metrics ==========")

    for group in sorted(group_dict.keys()):
        labels = np.array(group_dict[group]["labels"])
        scores = np.array(group_dict[group]["scores"])

        if len(np.unique(labels)) < 2:
            print(f"[{group}] Only one class — skipping (N={len(labels)})")
            continue

        auc = roc_auc_score(labels, scores)
        print(f"[{group}] AUC: {auc:.4f}  N={len(labels)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Video Model Evaluation")

    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--frames_per_video", type=int, default=24)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = VideoModel().to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    ckpt = torch.load(args.model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.model_path}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset = DeepfakedDataset(
        root_dir=args.root_dir,
        split=args.split,
        frames_per_video=args.frames_per_video,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"[INFO] Loaded {len(dataset)} samples (split='{args.split}')")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_labels = []
    all_scores = []

    # Optional grouping (dataset / attack / source)
    group_buckets = defaultdict(lambda: {"labels": [], "scores": []})

    with torch.no_grad():
        for batch in loader:
            # Support both formats
            if len(batch) == 3:
                frames, labels, meta = batch
            else:
                frames, labels = batch
                meta = None

            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_scores.extend(probs_np)
            all_labels.extend(labels_np)

            # -------- Grouping (if metadata exists) -------- #
            if meta is not None:
                for i in range(len(labels_np)):
                    group_name = meta[i].get("dataset", "unknown")
                    group_buckets[group_name]["labels"].append(labels_np[i])
                    group_buckets[group_name]["scores"].append(probs_np[i])

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    auc, best_thr = print_main_metrics(all_labels, all_scores)

    if len(group_buckets) > 0:
        print_group_metrics(group_buckets)

    # -------------------------------------------------------------------------
    # Save outputs (Fusion ready)
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "video_scores.npy"), all_scores)
    np.save(os.path.join(args.output_dir, "video_labels.npy"), all_labels)

    print(f"\n[INFO] Saved scores + labels → {args.output_dir}")


# =============================================================================

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()