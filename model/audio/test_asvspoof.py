# =============================================================================
# test_asvspoof.py
# Evaluation on ASVspoof 2019 LA eval set (index-based cache pipeline)
#
# Outputs:
#   - Overall AUC + EER
#   - Per-attack AUC + EER (each attack paired against full bonafide pool)
#   - asvspoof_scores.npy + asvspoof_labels.npy  (for fusion pipeline)
#
# Usage:
#   python test_asvspoof.py
#   python test_asvspoof.py --split val
#   python test_asvspoof.py --batch_size 16
# =============================================================================

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict

from audio_model import AudioResNet18
from data_loader import AudioDataset, collate_fn


# =============================================================================
# PATHS — edit these to match your setup
# =============================================================================

DEFAULT_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\audio_combined_best.pth"
)
DEFAULT_CACHE_ROOT = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Data\Audio\cache_wave"
)
DEFAULT_OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_outputs"
)


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def print_metrics(tag, labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class present — skipping  (N={len(labels)})")
        return
    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)
    print(f"  [{tag}] AUC: {auc:.4f}  EER: {eer:.4f}  N={len(labels)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA evaluation")
    parser.add_argument("--model_path",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache_root",  default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split",       default="test",
                        help="Cache split to evaluate: train / val / test")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.model_path}")

    # -------------------------------------------------------------------------
    # Dataset + DataLoader
    # -------------------------------------------------------------------------
    dataset = AudioDataset(
        cache_root=args.cache_root,
        splits=[args.split],
        augment=False,
        target_length=None
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn
    )

    print(f"[INFO] Loaded {len(dataset)} samples  split='{args.split}'")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x, y, _ in loader:          # collate_fn returns (x, y, dataset_str)
            x = x.to(device, non_blocking=True)

            logits = model(x).view(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(probs)
            all_labels.extend(y.numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------
    print("\n========== Overall Metrics ==========")
    print_metrics("ALL", all_labels, all_scores)

    # -------------------------------------------------------------------------
    # Per-attack metrics
    # Each attack contains only spoof samples, so we pair it against the full
    # bonafide pool — standard ASVspoof evaluation protocol.
    # -------------------------------------------------------------------------
    attack_buckets = defaultdict(lambda: {"labels": [], "scores": []})
    for i, sample_meta in enumerate(dataset.samples):
        attack = sample_meta.get("attack", "unknown")
        attack_buckets[attack]["labels"].append(all_labels[i])
        attack_buckets[attack]["scores"].append(all_scores[i])

    bonafide_labels = np.array(attack_buckets["bonafide"]["labels"])
    bonafide_scores = np.array(attack_buckets["bonafide"]["scores"])
    n_bonafide = len(bonafide_labels)

    print(f"\n========== Per-Attack Metrics (attack vs {n_bonafide} bonafide) ==========")
    for attack in sorted(attack_buckets.keys()):
        if attack == "bonafide":
            continue

        atk_labels = np.array(attack_buckets[attack]["labels"])
        atk_scores = np.array(attack_buckets[attack]["scores"])

        combined_labels = np.concatenate([bonafide_labels, atk_labels])
        combined_scores = np.concatenate([bonafide_scores, atk_scores])

        print_metrics(attack, combined_labels, combined_scores)

    # -------------------------------------------------------------------------
    # Save outputs for fusion pipeline
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "asvspoof_scores.npy"), all_scores)
    np.save(os.path.join(args.output_dir, "asvspoof_labels.npy"), all_labels)
    print(f"\n[INFO] Saved scores + labels → {args.output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
