import os
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
    r"\Models\video_finetuned\finetuned_best.pth"
)

CACHE_ROOT  = r"D:\FakeAVCache\Video"

SPLITS_JSON = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)

OUTPUT_DIR  = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video\eval_Fakeav_test_split\after_finetuning"
)

# FIX 1: Removed NPY_FILENAME constant.
# meta["file"] from the fixed cache index is already "abc123/frames.npy"
# (a full cache-root-relative path). Appending NPY_FILENAME again produced
# ".../abc123/frames.npy/frames.npy" which crashes on np.load().

NUM_FRAMES = 24  # MUST match cache & training


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

    auc     = roc_auc_score(labels, scores)
    eer     = compute_eer(labels, scores)
    opt_thr = find_optimal_threshold(labels, scores)

    preds  = (scores >= opt_thr).astype(int)
    acc    = accuracy_score(labels, preds)
    cm     = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=4)

    print(f"AUC                     : {auc:.4f}")
    print(f"EER                     : {eer:.4f}")
    print(f"Optimal Threshold       : {opt_thr:.4f}")
    print(f"Accuracy @ Optimal Thr  : {acc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    return auc, eer


def print_metrics(tag, labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class — skipping")
        return

    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)

    print(f"  [{tag:<22}]  AUC: {auc:.4f}  EER: {eer:.4f}  N={len(labels)}")


# =============================================================================
# DATASET
# =============================================================================

class TestSplitDataset(Dataset):

    def __init__(self, samples, cache_root, num_frames=24):
        self.samples    = samples
        self.cache_root = cache_root
        self.num_frames = num_frames

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        # FIX 1: meta["file"] is already the full relative path
        # e.g. "abc123/frames.npy" — resolve directly against cache_root.
        # Old code did: os.path.join(cache_root, meta["file"], NPY_FILENAME)
        # which produced ".../abc123/frames.npy/frames.npy" → crash.
        path = os.path.join(self.cache_root, meta["file"])
        clip = np.load(path)  # (T, H, W, C)

        T = clip.shape[0]

        if T >= self.num_frames:
            idxs = np.linspace(0, T - 1, self.num_frames).astype(int)
        else:
            idxs = np.concatenate([
                np.arange(T),
                np.full(self.num_frames - T, T - 1)
            ])

        clip = clip[idxs]

        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (0, 3, 1, 2))        # (T, C, H, W)
        clip = (clip - self.mean) / self.std
        clip = torch.from_numpy(clip).permute(1, 0, 2, 3)  # (C, T, H, W)

        label = torch.tensor(meta["label"], dtype=torch.float32)

        return clip, label, idx


def collate_fn(batch):
    xs  = torch.stack([b[0] for b in batch])
    ys  = torch.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return xs, ys, ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache_root",  default=CACHE_ROOT)
    parser.add_argument("--splits_json", default=SPLITS_JSON)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--batch_size",  type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_frames",  type=int, default=NUM_FRAMES)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[INFO] Device : {device}")

    # -------------------------------------------------------------------------
    # Load splits + index
    # -------------------------------------------------------------------------
    print("[INFO] Loading splits...")
    with open(args.splits_json) as f:
        splits = json.load(f)

    print("[INFO] Loading index...")
    with open(os.path.join(args.cache_root, "video_index.json")) as f:
        index = json.load(f)

    test_indices = splits["test_indices"]
    test_samples = [index[i] for i in test_indices]

    print(f"  Test samples : {len(test_samples)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Loading model: {args.model_path}")

    model = VideoModel(num_classes=2, pretrained=False, dropout=0.3).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  val_auc: {ckpt.get('val_auc', '?')}")

    # -------------------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------------------
    dataset = TestSplitDataset(test_samples, args.cache_root, args.num_frames)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_labels   = []
    all_scores   = []
    all_indices  = []

    with torch.no_grad():
        for x, y, ids in tqdm(loader, desc="Inference"):

            x = x.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
            else:
                logits = model(x)

            if logits.ndim == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits.view(-1))

            all_scores.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_indices.extend(ids)

    all_labels  = np.array(all_labels)
    all_scores  = np.array(all_scores)
    all_indices = np.array(all_indices)

    print(f"[INFO] Done — {len(all_labels)} samples")

    # -------------------------------------------------------------------------
    # FIX 3: Save file_ids alongside scores so fusion layer can join on
    # file_id instead of positional index.
    # audio eval already saves audio_test_file_ids.npy — both sides must
    # use the same key for the fusion join to work.
    # -------------------------------------------------------------------------
    all_file_ids = np.array([test_samples[i]["file_id"] for i in all_indices])

    np.save(os.path.join(args.output_dir, "finetuned_video_scores.npy"),   all_scores)
    np.save(os.path.join(args.output_dir, "finetuned_video_labels.npy"),   all_labels)
    np.save(os.path.join(args.output_dir, "finetuned_video_indices.npy"),  all_indices)
    np.save(os.path.join(args.output_dir, "finetuned_video_file_ids.npy"), all_file_ids)  # FIX 3

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    print("\n========== OVERALL ==========")
    auc, eer = print_full_metrics(all_labels, all_scores)

    # -------------------------------------------------------------------------
    # Per-attack
    # -------------------------------------------------------------------------
    real_labels, real_scores = [], []
    attack_buckets = defaultdict(lambda: {"labels": [], "scores": []})

    for pos, dataset_idx in enumerate(all_indices):
        label = all_labels[pos]
        score = all_scores[pos]
        meta  = test_samples[dataset_idx]

        if label == 0:
            real_labels.append(label)
            real_scores.append(score)
        else:
            atk = meta["attack"]
            attack_buckets[atk]["labels"].append(label)
            attack_buckets[atk]["scores"].append(score)

    real_labels = np.array(real_labels)
    real_scores = np.array(real_scores)

    print("\n========== PER-ATTACK ==========")
    for atk in sorted(attack_buckets.keys()):
        l = np.concatenate([real_labels, attack_buckets[atk]["labels"]])
        s = np.concatenate([real_scores, attack_buckets[atk]["scores"]])
        print_metrics(atk, l, s)

    print("\n[INFO] Finished.")
    print(f"  AUC: {auc:.4f}  EER: {eer:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()