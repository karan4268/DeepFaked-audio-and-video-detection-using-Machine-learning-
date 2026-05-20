# =============================================================================
# test_FakeAVCelebs.py
# Evaluation on FakeAVCeleb video cache (index-based,cache built by "cache_fakeav_video.py")
#
# Cache lives at D:\FakeAVCache\Video
# index.json fields: file_id, file (md5 hash), label, speaker, attack, dataset
#
# Attack values in this cache:
#   bonafide | faceswap-wav2lip | fsgan-wav2lip | rtvc | wav2lip
#
# Loading priority:
#   1. frames.npy  (fast — run convert_cache_to_npy.py first)
# 
# Outputs:
#   - Overall AUC, Optimal Threshold, Accuracy, Confusion Matrix, Report
#   - Per-attack  AUC + EER  (each method's fakes vs full real pool)
#   - Per-speaker AUC + EER  (each speaker's fakes vs full real pool)
#   - fakeavceleb_video_scores.npy + fakeavceleb_video_labels.npy  (for fusion pipeline)
# =============================================================================

import os
import argparse
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, confusion_matrix, classification_report
)
from collections import defaultdict
from tqdm import tqdm

from video_model import VideoModel


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_combined\combined_train_best.pth"
)
DEFAULT_CACHE_ROOT = r"D:\FakeAVCache\Video"
DEFAULT_OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video\eval_outputs\FakeAVCeleb"
)

FRAMES_PER_VIDEO = 24
NPY_FILENAME     = "frames.npy"


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def find_optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return thresholds[np.argmax(tpr - fpr)]


def print_full_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    print(f"[DEBUG] labels shape: {labels.shape}  scores shape: {scores.shape}")
    assert labels.shape == scores.shape, (
        f"Shape mismatch — labels: {labels.shape}, scores: {scores.shape}"
    )

    auc       = roc_auc_score(labels, scores)
    opt_thr   = find_optimal_threshold(labels, scores)
    preds_opt = (scores >= opt_thr).astype(int)
    preds_05  = (scores >= 0.5).astype(int)
    acc_opt   = accuracy_score(labels, preds_opt)
    acc_05    = accuracy_score(labels, preds_05)
    cm        = confusion_matrix(labels, preds_opt)
    report    = classification_report(labels, preds_opt, digits=4)

    print(f"AUC                     : {auc:.4f}")
    print(f"Optimal Threshold       : {opt_thr:.4f}")
    print(f"Accuracy @ Optimal Thr  : {acc_opt:.4f}")
    print(f"Accuracy @ 0.5 Thr      : {acc_05:.4f}")
    print(f"\nConfusion Matrix (Optimal):\n{cm}")
    print(f"\nClassification Report:\n{report}")


def print_metrics(tag, labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class present — skipping  (N={len(labels)})")
        return
    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)
    print(f"  [{tag}]  AUC: {auc:.4f}  EER: {eer:.4f}  N={len(labels)}")


# =============================================================================
# DATASET (Convert the png-based cache to a PyTorch Dataset that loads frames.npy for speed)
# =============================================================================

class FakeAVCelebVideoDataset(Dataset):

    def __init__(self, cache_root):
        index_path = os.path.join(cache_root, "index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"index.json not found: {index_path}")

        with open(index_path, "r") as f:
            self.samples = json.load(f)

        self.cache_root = cache_root

        n_real = sum(1 for s in self.samples if s["label"] == 0)
        n_fake = sum(1 for s in self.samples if s["label"] == 1)

        attack_counts = defaultdict(int)
        for s in self.samples:
            if s["label"] == 1:
                attack_counts[s["attack"]] += 1

        # Strict check
        missing = 0
        for s in self.samples[:100]:
            path = os.path.join(cache_root, s["file"], NPY_FILENAME)
            if not os.path.isfile(path):
                missing += 1

        if missing > 0:
            raise RuntimeError(
                f"[ERROR] {missing}/100 samples missing frames.npy. Run conversion first."
            )

        print(f"[INFO] index.json — real: {n_real}  fake: {n_fake}  total: {len(self.samples)}")
        print("[INFO] Loading mode : NPY (strict, optimized)")
        print("[INFO] Attack breakdown:")
        for atk, cnt in sorted(attack_counts.items()):
            print(f"         {atk:<22} : {cnt}")

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, frame_dir):
        clip = np.load(os.path.join(frame_dir, NPY_FILENAME))  # (T,H,W,C)

        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (0, 3, 1, 2))  # (T,C,H,W)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)

        clip = (clip - mean) / std

        return torch.from_numpy(clip)

    def __getitem__(self, idx):
        meta      = self.samples[idx]
        frame_dir = os.path.join(self.cache_root, meta["file"])
        label     = meta["label"]

        npy_path = os.path.join(frame_dir, NPY_FILENAME)

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Missing frames.npy in {frame_dir}")

        frames = self._load_npy(frame_dir)
        frames = frames.permute(1, 0, 2, 3)  # [C,T,H,W]

        return (
            frames,
            torch.tensor(label, dtype=torch.float32),
            idx
        )


def collate_fn(batch):
    xs  = torch.stack([b[0] for b in batch])
    ys  = torch.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return xs, ys, ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache_root",  default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size",  type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = VideoModel().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpointfrom:{args.model_path}")

    # Probe model output shape to determine if it's 2-class (softmax) or single-logit (sigmoid)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, FRAMES_PER_VIDEO, 224, 224).to(device)
        try:
            probe = model(dummy)
            print(f"[INFO] Model output shape (batch=1): {probe.shape}")
            two_class_output = (probe.ndim == 2 and probe.shape[1] == 2)
        except Exception as e:
            print(f"[WARN] Could not probe model shape: {e}. Assuming single-logit output.")
            two_class_output = False

    if two_class_output:
        print("[INFO] Detected 2-class output → using softmax(:,1) for fake probability")
    else:
        print("[INFO] Detected single-logit output → using sigmoid")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    dataset = FakeAVCelebVideoDataset(args.cache_root)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn
    )

    print(f"[INFO] Loaded {len(dataset)} samples")

    # -------------------------------------------------------------------------
    # Inference + Progress Bar (add this because the dataset is large and take hr for inference)
    # -------------------------------------------------------------------------
    all_labels, all_scores, all_indices = [], [], []

    start = time.time()

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Inference", ncols=100)

        for i, (x, y, ids) in enumerate(pbar):
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(x)                          # [B,2] or [B,1] or [B]

            # -----------------------------------------------------------------
            # FIX: correctly extract per-sample fake probability
            # -----------------------------------------------------------------
            if two_class_output:
                # model outputs [B, 2] class logits → softmax, take fake class (col 1)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                # model outputs [B, 1] or [B] single logit → sigmoid
                probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            # -----------------------------------------------------------------

            all_scores.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
            all_indices.extend(ids)

            # live stats
            elapsed = time.time() - start
            done = min((i + 1) * args.batch_size, len(dataset))
            speed = done / elapsed if elapsed > 0 else 0

            pbar.set_postfix({
                "samples": f"{done}/{len(dataset)}",
                "speed": f"{speed:.1f}/s"
            })

    all_labels  = np.array(all_labels)
    all_scores  = np.array(all_scores)
    all_indices = np.array(all_indices)

    # Sanity check before any metrics
    assert len(all_labels) == len(all_scores) == len(all_indices), (
        f"[ERROR] Length mismatch — labels:{len(all_labels)}  "
        f"scores:{len(all_scores)}  indices:{len(all_indices)}"
    )
    print(f"[INFO] Inference complete — {len(all_labels)} samples collected")

    # -------------------------------------------------------------------------
    # Save raw outputs immediately so metrics can be recomputed without re-inference(can be removed but good for safety so that if metrics code has bugs/crashes,does not require re-running the long inference step) learned the hard way
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "raw_scores.npy"),  all_scores)
    np.save(os.path.join(args.output_dir, "raw_labels.npy"),  all_labels)
    np.save(os.path.join(args.output_dir, "raw_indices.npy"), all_indices)
    print(f"[INFO] Raw outputs saved → {args.output_dir}")
    print("[INFO] If metrics crash, reload raw_*.npy and recompute without re-inference")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    print("\n========== Overall Metrics ==========")
    print_full_metrics(all_labels, all_scores)

    real_pool_labels = []
    real_pool_scores = []

    for pos, idx in enumerate(all_indices):
        if all_labels[pos] == 0:
            real_pool_labels.append(all_labels[pos])
            real_pool_scores.append(all_scores[pos])

    real_pool_labels = np.array(real_pool_labels)
    real_pool_scores = np.array(real_pool_scores)
    n_real = len(real_pool_labels)

    # Per-attack
    attack_buckets = defaultdict(lambda: {"labels": [], "scores": []})

    for pos, idx in enumerate(all_indices):
        if all_labels[pos] == 1:
            atk = dataset.samples[idx]["attack"]
            attack_buckets[atk]["labels"].append(all_labels[pos])
            attack_buckets[atk]["scores"].append(all_scores[pos])

    print(f"\n========== Per-Attack Metrics (vs {n_real} real) ==========")
    for atk in sorted(attack_buckets.keys()):
        l = np.concatenate([real_pool_labels, attack_buckets[atk]["labels"]])
        s = np.concatenate([real_pool_scores, attack_buckets[atk]["scores"]])
        print_metrics(atk, l, s)

    # Per-speaker
    speaker_buckets = defaultdict(lambda: {"labels": [], "scores": []})

    for pos, idx in enumerate(all_indices):
        if all_labels[pos] == 1:
            spk = dataset.samples[idx]["speaker"]
            speaker_buckets[spk]["labels"].append(all_labels[pos])
            speaker_buckets[spk]["scores"].append(all_scores[pos])

    print(f"\n========== Per-Speaker Metrics ==========")
    for spk in sorted(speaker_buckets.keys()):
        l = np.concatenate([real_pool_labels, speaker_buckets[spk]["labels"]])
        s = np.concatenate([real_pool_scores, speaker_buckets[spk]["scores"]])
        print_metrics(spk, l, s)

    # Save fusion outputs
    np.save(os.path.join(args.output_dir, "fakeavceleb_video_scores.npy"), all_scores)
    np.save(os.path.join(args.output_dir, "fakeavceleb_video_labels.npy"), all_labels) # npy format usefull for fusionn later
    print(f"\n[INFO] Saved fusion outputs → {args.output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()