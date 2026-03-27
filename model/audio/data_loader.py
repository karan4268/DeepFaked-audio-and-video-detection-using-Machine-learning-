
# data_loader.py
# ------------------------------------------------------------
# ASVspoof Dataset (Dynamic Waveform + Augmentation + Batch Padding)
# ------------------------------------------------------------
 
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from Preprocessing.extractor import compute_log_mel


# ------------------------------------------------------------
# AUGMENTATIONS
# ------------------------------------------------------------

def random_crop(audio, target_len=None):
    if target_len is None or len(audio) <= target_len:
        return audio
    start = np.random.randint(0, len(audio) - target_len)
    return audio[start:start + target_len]


def add_noise(audio):
    noise = np.random.randn(len(audio))
    scale = np.random.uniform(0.001, 0.01)
    return audio + scale * noise


def random_gain(audio):
    gain = np.random.uniform(0.8, 1.2)
    return audio * gain


def time_shift(audio):
    shift = np.random.randint(-2000, 2000)
    return np.roll(audio, shift)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

class ASVSpoofDataset(Dataset):
    def __init__(
        self,
        cache_root,
        split,
        augment=False,
        target_length=None,
        aug_prob=0.5,
        allowed_attacks=None  
    ):
        self.cache_root    = cache_root
        self.split         = split
        self.augment       = augment
        self.target_length = target_length
        self.aug_prob      = aug_prob
        self.allowed_attacks = allowed_attacks

        self.samples = []

        index_path = os.path.join(cache_root, split, "index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing index: {index_path}")

        with open(index_path, "r") as f:
            index_data = json.load(f)

        for entry in index_data:
            attack = entry.get("attack", "unknown")

            # FILTER ATTACKS
            if self.allowed_attacks is not None:
                if attack not in self.allowed_attacks:
                    continue

            label     = entry["label"]
            file_name = entry["file"]

            label_dir = "real" if label == 0 else "fake"
            path      = os.path.join(cache_root, split, label_dir, file_name)

            if os.path.isfile(path):
                self.samples.append({
                    "path": path,
                    "label": label,
                    "attack": attack,
                    "speaker": entry.get("speaker", None)
                })

        self.labels = [s["label"] for s in self.samples]

        print(f"[INFO] {split}: Loaded {len(self.samples)} samples "
              f"(real={self.labels.count(0)}, spoof={self.labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path   = sample["path"]
        label  = sample["label"]

        # -------------------------------
        # LOAD
        # -------------------------------
        audio = np.load(path).astype(np.float32) / 32768.0

        if np.isnan(audio).any():
            raise ValueError(f"NaNs found in audio: {path}")

        # -------------------------------
        # CROP
        # -------------------------------
        if self.target_length:
            audio = random_crop(audio, self.target_length)

        # -------------------------------
        # AUGMENT
        # -------------------------------
        if self.augment and np.random.rand() < self.aug_prob:
            if np.random.rand() < 0.5:
                audio = add_noise(audio)
            if np.random.rand() < 0.5:
                audio = random_gain(audio)
            if np.random.rand() < 0.3:
                audio = time_shift(audio)

        # -------------------------------
        # FEATURE
        # -------------------------------
        log_mel = compute_log_mel(audio)

        if np.isnan(log_mel).any():
            raise ValueError(f"NaNs in log-mel: {path}")

        log_mel = np.nan_to_num(log_mel)

        # -------------------------------
        # FIX LENGTH
        # -------------------------------
        MAX_FRAMES = 400
        T = log_mel.shape[-1]

        if T > MAX_FRAMES:
            if self.augment:
                start = np.random.randint(0, T - MAX_FRAMES)
            else:
                start = (T - MAX_FRAMES) // 2
            log_mel = log_mel[:, start:start + MAX_FRAMES]

        elif T < MAX_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - T)))

        return (
            torch.from_numpy(log_mel).unsqueeze(0).float(),
            torch.tensor(label, dtype=torch.float32)
        )

# ------------------------------------------------------------
# COLLATE
# ------------------------------------------------------------

def collate_fn(batch):
    batch_x = [x for x, _ in batch]
    batch_y = [y for _, y in batch]
    return torch.stack(batch_x), torch.stack(batch_y)

