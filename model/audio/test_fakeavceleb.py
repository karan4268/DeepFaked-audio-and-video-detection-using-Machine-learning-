# =============================================================================
# test_fakeavceleb_FIXED.py
#
# FIXED VERSION:
#   - Preserves file_id during inference
#   - Safe for multi-worker DataLoader
#   - Produces fusion-ready aligned outputs
#
# Outputs:
#   - fakeavceleb_scores.npy
#   - fakeavceleb_labels.npy
#   - fakeavceleb_file_ids.npy   <-- NEW (CRITICAL)
# =============================================================================

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict

from audio_model import AudioResNet18
from data_loader import AudioDataset


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\audio_combined_best.pth"
DEFAULT_CACHE_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
DEFAULT_OUTPUT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\eval_fakeavceleb"


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
# CUSTOM COLLATE (WITH FILE_ID)
# =============================================================================

def collate_fn_with_ids(batch):
    x, y, ds, fid = zip(*batch)

    x = torch.stack(x)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, ds, fid


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache_root",  default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="fakeav_test")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}| {device_name}")

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[INFO] Loaded checkpoint")

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------
    dataset = AudioDataset(
        cache_root=args.cache_root,
        splits=[args.split],
        augment=False,
        target_length=None
    )

    # Keep only FakeAVCeleb
    dataset.samples = [s for s in dataset.samples if s.get("dataset") == "fakeav"]

    if len(dataset) == 0:
        raise RuntimeError("No FakeAVCeleb samples found.")

    print(f"[INFO] Samples loaded: {len(dataset)}")

    # -------------------------------------------------------------------------
    # DATALOADER
    # -------------------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn_with_ids
    )

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    all_scores   = []
    all_labels   = []
    all_datasets = []
    all_file_ids = []

    with torch.no_grad():
        for x, y, ds, fid in loader:
            x = x.to(device, non_blocking=True)

            logits = model(x).view(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(probs)
            all_labels.extend(y.numpy())
            all_datasets.extend(ds)
            all_file_ids.extend(fid)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_file_ids = np.array(all_file_ids)

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    print("\n========== Overall Metrics ==========")
    print_metrics("ALL", all_labels, all_scores)

    # -------------------------------------------------------------------------
    # PER-SPEAKER
    # -------------------------------------------------------------------------
    speaker_buckets  = defaultdict(lambda: {"labels": [], "scores": []})
    real_pool_labels = []
    real_pool_scores = []

    for i, sample_meta in enumerate(dataset.samples):
        label   = all_labels[i]
        score   = all_scores[i]
        speaker = sample_meta.get("speaker", "unknown")

        if label == 0:
            real_pool_labels.append(label)
            real_pool_scores.append(score)
        else:
            speaker_buckets[speaker]["labels"].append(label)
            speaker_buckets[speaker]["scores"].append(score)

    real_pool_labels = np.array(real_pool_labels)
    real_pool_scores = np.array(real_pool_scores)

    print("\n========== Per-Speaker Metrics ==========")
    for speaker in sorted(speaker_buckets.keys()):
        spk_labels = np.array(speaker_buckets[speaker]["labels"])
        spk_scores = np.array(speaker_buckets[speaker]["scores"])

        combined_labels = np.concatenate([real_pool_labels, spk_labels])
        combined_scores = np.concatenate([real_pool_scores, spk_scores])

        print_metrics(speaker, combined_labels, combined_scores)

    # -------------------------------------------------------------------------
    # SAVE (FUSION READY)
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "fakeavceleb_scores.npy"), all_scores)
    np.save(os.path.join(args.output_dir, "fakeavceleb_labels.npy"), all_labels)
    np.save(os.path.join(args.output_dir, "fakeavceleb_file_ids.npy"), all_file_ids)

    print("\n[INFO] Saved:")
    print("  - scores")
    print("  - labels")
    print("  - file_ids  ✅ (critical for alignment)")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()