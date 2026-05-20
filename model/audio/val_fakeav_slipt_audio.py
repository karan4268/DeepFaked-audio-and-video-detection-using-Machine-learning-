# =============================================================================
# val_audio_on_split.py
# Runs audio model inference on the VAL split of FakeAVCeleb.
# Outputs val scores + file_ids used for oracle-free alpha selection in fusion.
#
# NOTE: Audio model was trained on ASVspoof + ITW only — zero-shot on FakeAV.
#       This val pass is purely a forward pass, no fine-tuning involved.
# =============================================================================

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

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

AUDIO_INDEX = os.path.join(
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Data\Audio\cache_wave",
    "fakeav_test", "audio_index.json"
)

# Source of truth for val file_ids — use video index (same splits.json)
VIDEO_INDEX = r"D:\FakeAVCache\Video\video_index.json"

SPLITS_JSON = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)

OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_val_for_alpha"
)


# =============================================================================
# DATASET  (same as FakeAVAudioDataset in test script — no augmentation)
# =============================================================================

class FakeAVAudioDataset(Dataset):

    def __init__(self, samples, cache_root):
        self.samples    = samples
        self.cache_root = cache_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # entry["file"] is already cache-root-relative
        path = os.path.join(self.cache_root, entry["file"])

        try:
            audio = np.load(path).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e} — returning zeros")
            audio = np.zeros(16000, dtype=np.float32)

        if np.isnan(audio).any():
            print(f"[WARNING] NaN in {path} — zero-filling")
            audio = np.zeros_like(audio)

        log_mel = compute_log_mel(audio)
        log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

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
    # Load splits + video index (source of truth for val file_ids)
    # -------------------------------------------------------------------------
    print("[INFO] Loading splits + video index...")
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    with open(VIDEO_INDEX) as f:
        video_index = json.load(f)

    val_indices  = splits["val_indices"]
    val_file_ids = set(video_index[i]["file_id"] for i in val_indices)

    print(f"  Val file_ids from video index: {len(val_file_ids)}")

    # -------------------------------------------------------------------------
    # Load audio index + filter to val split
    # -------------------------------------------------------------------------
    print("[INFO] Loading audio index...")
    with open(AUDIO_INDEX) as f:
        audio_index = json.load(f)

    audio_samples = [
        e for e in audio_index
        if e.get("file_id") in val_file_ids
    ]

    print(f"  Audio samples after val-split filter: {len(audio_samples)}")

    if len(audio_samples) == 0:
        raise RuntimeError(
            "No audio samples matched val_file_ids. "
            "Check that audio cache and video index share the same file_id convention."
        )

    real_count = sum(1 for e in audio_samples if e["label"] == 0)
    fake_count = sum(1 for e in audio_samples if e["label"] == 1)
    print(f"  Real: {real_count}  Fake: {fake_count}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[INFO] Audio model loaded")

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

    labels = np.array(all_labels)
    scores = np.array(all_scores)

    print(f"\n[INFO] Done — {len(labels)} samples")
    print(f"[INFO] Val AUC (audio): {roc_auc_score(labels, scores):.4f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "audio_val_scores.npy"),   scores)
    np.save(os.path.join(OUTPUT_DIR, "audio_val_labels.npy"),   labels)
    np.save(os.path.join(OUTPUT_DIR, "audio_val_file_ids.npy"), np.array(all_file_ids))

    print(f"[INFO] Saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()