# =============================================================================
# test_audio_on_split.py  (FIXED)
#
# FIXES vs previous version:
#
#   [FIX 1] Bypassed AudioDataset path construction for FakeAV entries.
#           AudioDataset builds: cache_root/split/label_dir/file_name
#           But audio_index.json now stores file_name as a full relative path
#           e.g. "fakeav_test/real/abc123.npy", so the old construction
#           produced "cache_root/fakeav_test/real/fakeav_test/real/abc123.npy".
#           Every os.path.isfile() check failed silently → empty dataset.
#           Fix: load fakeav audio directly from audio_index.json, resolve
#           path as os.path.join(AUDIO_CACHE_ROOT, entry["file"]) only.
#
#   [FIX 2] audio_test_file_ids.npy was already saved — no change needed.
#           Video eval (separately fixed) now also saves file_ids so the
#           fusion layer can join both sides on file_id correctly.
#
# Outputs:
#   audio_test_scores.npy
#   audio_test_labels.npy
#   audio_test_file_ids.npy   <-- fusion join key
# =============================================================================

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from data_loader import collate_fn
from Preprocessing.extractor import compute_log_mel


# =============================================================================
# PATHS
# =============================================================================

MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\audio_combined_best.pth"
)

AUDIO_CACHE_ROOT = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Data\Audio\cache_wave"
)

VIDEO_INDEX = r"D:\FakeAVCache\Video\video_index.json"

AUDIO_INDEX = os.path.join(AUDIO_CACHE_ROOT, "fakeav_test", "audio_index.json")

SPLITS_JSON = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)

OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_fakeavceleb_test_split"
)


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


# =============================================================================
# FIX 1: Direct FakeAV audio dataset
# Bypasses AudioDataset path construction which prepends cache_root/split/label_dir
# on top of a file field that already contains that prefix.
# Path is resolved as: os.path.join(AUDIO_CACHE_ROOT, entry["file"]) only.
# =============================================================================

class FakeAVAudioDataset(Dataset):

    def __init__(self, samples, cache_root):
        self.samples    = samples
        self.cache_root = cache_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # FIX 1: direct resolution — no split/label_dir injection
        path = os.path.join(self.cache_root, entry["file"])

        try:
            audio = np.load(path).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e} — returning zeros")
            audio = np.zeros(16000, dtype=np.float32)

        if np.isnan(audio).any():
            print(f"[WARNING] NaN in {path} — zero-filling")
            audio = np.zeros_like(audio)

        # Feature extraction
        log_mel = compute_log_mel(audio)
        log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad / crop to MAX_FRAMES
        MAX_FRAMES = 400
        T = log_mel.shape[-1]

        if T > MAX_FRAMES:
            start   = (T - MAX_FRAMES) // 2
            log_mel = log_mel[:, start:start + MAX_FRAMES]
        elif T < MAX_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - T)))

        return (
            torch.from_numpy(log_mel).unsqueeze(0).float(),
            torch.tensor(entry["label"], dtype=torch.float32),
            entry.get("dataset", "fakeav"),
            entry["file_id"],
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -------------------------------------------------------------------------
    # Load splits + video index (source of truth for test file_ids)
    # -------------------------------------------------------------------------
    print("[INFO] Loading splits + index...")

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    with open(VIDEO_INDEX) as f:
        video_index = json.load(f)

    test_indices  = splits["test_indices"]
    test_file_ids = set(video_index[i]["file_id"] for i in test_indices)

    print(f"[INFO] Test file_ids from video index: {len(test_file_ids)}")

    # -------------------------------------------------------------------------
    # Load audio index + filter to test split
    # -------------------------------------------------------------------------
    print("[INFO] Loading audio index...")

    with open(AUDIO_INDEX) as f:
        audio_index = json.load(f)

    # FIX 1: filter directly from audio_index — no AudioDataset path logic
    audio_samples = [
        e for e in audio_index
        if e.get("file_id") in test_file_ids
    ]

    print(f"[INFO] Audio samples after test-split filter: {len(audio_samples)}")

    if len(audio_samples) == 0:
        raise RuntimeError(
            "No audio samples matched test_file_ids. "
            "Check that audio cache and video index share the same file_id convention."
        )

    real_count = sum(1 for e in audio_samples if e["label"] == 0)
    fake_count = sum(1 for e in audio_samples if e["label"] == 1)
    print(f"[INFO] Real: {real_count}  Fake: {fake_count}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[INFO] Model loaded")

    # -------------------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------------------
    dataset = FakeAVAudioDataset(audio_samples, AUDIO_CACHE_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_scores   = []
    all_labels   = []
    all_file_ids = []

    with torch.no_grad():
        for x, y, ds, fid in loader:
            x = x.to(device)

            logits = model(x).view(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(probs)
            all_labels.extend(y.cpu().numpy())
            all_file_ids.extend(fid)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    labels = np.array(all_labels)
    scores = np.array(all_scores)

    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)

    print("\n========== AUDIO TEST (SPLIT-ALIGNED) ==========")
    print(f"AUC : {auc:.4f}")
    print(f"EER : {eer:.4f}")
    print(f"N   : {len(labels)}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "audio_test_scores.npy"),   scores)
    np.save(os.path.join(OUTPUT_DIR, "audio_test_labels.npy"),   labels)
    np.save(os.path.join(OUTPUT_DIR, "audio_test_file_ids.npy"), np.array(all_file_ids))

    print(f"\n[INFO] Saved outputs → {OUTPUT_DIR}")


# =============================================================================

if __name__ == "__main__":
    main()