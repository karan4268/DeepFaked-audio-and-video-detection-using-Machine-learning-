# data_loader.py
# ------------------------------------------------------------
# Unified Dataset (ASVspoof + ITW)
#
# FIXES vs previous version:
#   [FIX 1] allowed_attacks filter is now dataset-aware.
#           Previously the filter was applied uniformly, so passing
#           ASV_TRAIN_ATTACKS would silently drop ALL ITW fake samples
#           because ITW attacks are labelled "itw_{speaker}", not A01–A19.
#           Now the filter only applies to asvspoof entries; ITW entries
#           always pass through regardless of allowed_attacks.
#
#   [FIX 2] Replaced the ValueError raise on NaN audio with a silent
#           zero-fill fallback. A single corrupt file in a 30k-sample
#           ITW cache was crashing the entire DataLoader worker with no
#           recovery path. Now it zero-fills and prints a warning so
#           training continues.
#
# IMPROVEMENTS:
#   [NEW 1] SpecAugment applied after log-mel computation during training.
#           Masks random frequency bands and time steps independently.
#           This is empirically the strongest regularizer for cross-domain
#           generalization in speech models (Park et al., 2019).
#
#   [NEW 2] aug_prob raised to 0.7 (from default) and SpecAugment gated
#           separately via spec_aug_prob so it can be tuned independently
#           of waveform augmentations.
# ------------------------------------------------------------

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from Preprocessing.extractor import compute_log_mel


# ------------------------------------------------------------
# WAVEFORM AUGMENTATIONS
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
# SPECAUGMENT  [NEW 1]
# Frequency masking + time masking on the log-mel spectrogram.
# Applied AFTER feature extraction so waveform augmentations
# and SpecAugment operate on independent axes.
#
# freq_mask  – max number of mel bins to zero out (default 20/128 = 15%)
# time_mask  – max number of time frames to zero out (default 40/400 = 10%)
# n_freq     – number of independent frequency masks
# n_time     – number of independent time masks
# ------------------------------------------------------------

def spec_augment(
    log_mel: np.ndarray,
    freq_mask: int = 20,
    time_mask: int = 40,
    n_freq: int = 2,
    n_time: int = 2
) -> np.ndarray:
    F, T = log_mel.shape
    out = log_mel.copy()

    for _ in range(n_freq):
        f = np.random.randint(1, max(2, freq_mask))
        f0 = np.random.randint(0, max(1, F - f))
        out[f0:f0 + f, :] = 0.0

    for _ in range(n_time):
        t = np.random.randint(1, max(2, time_mask))
        t0 = np.random.randint(0, max(1, T - t))
        out[:, t0:t0 + t] = 0.0

    return out


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(
        self,
        cache_root,
        splits,                      # list: ["train", "itw_train"]
        augment=False,
        target_length=None,
        aug_prob=0.7,
        spec_aug_prob=0.5,           #  independent SpecAugment gate
        allowed_attacks=None         # only filters asvspoof entries 
    ):
        self.cache_root    = cache_root
        self.splits        = splits
        self.augment       = augment
        self.target_length = target_length
        self.aug_prob      = aug_prob
        self.spec_aug_prob = spec_aug_prob
        self.allowed_attacks = allowed_attacks

        self.samples = []

        # ----------------------------------------------------
        # LOAD MULTIPLE SPLITS
        # ----------------------------------------------------
        for split in splits:
            index_path = os.path.join(cache_root, split, "index.json") # index.json when using for training on avspoof and itw datasets
            print(f"[INFO] Loading index for split {split} from {index_path} training cache")
            if not os.path.isfile(index_path): # if statements handels the case when index.json is not present in the cache (like in fakeavcelebs) and looks for audio_index.json instead.
                index_path = os.path.join(cache_root, split, "audio_index.json")
                print(f"[INFO] using audio_index.json for split {split} in fakeavtest cache")
            if not os.path.isfile(index_path):
                raise FileNotFoundError(f"No index found in: {os.path.join(cache_root, split)}") #audio_index.json is only for testing on fakeavcelebs.

            if not os.path.isfile(index_path):
                raise FileNotFoundError(f"Missing index: {index_path}")

            with open(index_path, "r") as f:
                index_data = json.load(f)

            for entry in index_data:
                attack  = entry.get("attack", "unknown")
                dataset = entry.get("dataset", "unknown")

                # [FIX 1] Attack filter applies ONLY to asvspoof samples.
                # ITW attacks are named "itw_{speaker}" / "bonafide" and would
                # all be dropped if the filter were applied uniformly with an
                # ASV-specific allowlist like ["bonafide", "A01", ...].
                if self.allowed_attacks is not None:
                    if dataset == "asvspoof" and attack not in self.allowed_attacks:
                        continue
                    # ITW samples: always keep, filter has no meaning for them

                label     = entry["label"]
                file_name = entry["file"]
                file_id = entry.get("file_id", file_name)

                if os.sep in file_name or "/" in file_name:
                    # file already contains subdirectory (e.g. fakeav_test\real\abc.npy)used for fakeavcelebs test cache
                    path = os.path.join(cache_root, file_name)
                else:
                    # plain filename (ASVspoof / ITW style) used for training caches
                    label_dir = "real" if label == 0 else "fake"
                    path = os.path.join(cache_root, split, label_dir, file_name)

                if os.path.isfile(path):
                    self.samples.append({
                            "path": path,
                            "label": label,
                            "attack": attack,
                            "speaker": entry.get("speaker", None),
                            "dataset": dataset,
                            "file_id": file_id  
                        })

        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------
        self.labels   = [s["label"]   for s in self.samples]
        self.datasets = [s["dataset"] for s in self.samples]

        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        asv_count  = self.datasets.count("asvspoof")
        itw_count  = self.datasets.count("itw")

        print(f"[INFO] Loaded {len(self.samples)} samples from {splits}")
        print(f"[INFO] Real: {real_count} | Fake: {fake_count}")
        print(f"[INFO] ASVspoof: {asv_count} | ITW: {itw_count}(will show 0 if fakeavcelebs is used for testing [ignore])")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded from {splits}. "
                "Check cache paths and allowed_attacks filter."
            )

        if real_count == 0 or fake_count == 0:
            raise RuntimeError(
                f"Only one class present after filtering "
                f"(real={real_count}, fake={fake_count}). "
                "Check allowed_attacks — bonafide must be included."
            )

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        path    = sample["path"]
        label   = sample["label"]
        dataset = sample["dataset"]

        # -------------------------------
        # LOAD
        # -------------------------------
        try:
            audio = np.load(path).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e} — returning zeros")
            audio = np.zeros(16000, dtype=np.float32)

        # [FIX 2] Silent NaN recovery instead of crashing the DataLoader worker.
        if np.isnan(audio).any():
            print(f"[WARNING] NaN audio at {path} — zero-filling")
            audio = np.zeros_like(audio)

        # -------------------------------
        # CROP
        # -------------------------------
        if self.target_length:
            audio = random_crop(audio, self.target_length)

        # -------------------------------
        # WAVEFORM AUGMENTATIONS
        # -------------------------------
        if self.augment and np.random.rand() < self.aug_prob:
            if np.random.rand() < 0.5:
                audio = add_noise(audio)
            if np.random.rand() < 0.5:
                audio = random_gain(audio)
            if np.random.rand() < 0.3:
                audio = time_shift(audio)

        # -------------------------------
        # FEATURE EXTRACTION
        # -------------------------------
        log_mel = compute_log_mel(audio)
        log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

        # -------------------------------
        # SPECAUGMENT  [NEW 1]
        # Applied after feature extraction, independently of waveform augs.
        # -------------------------------
        if self.augment and np.random.rand() < self.spec_aug_prob:
            log_mel = spec_augment(log_mel)

        # -------------------------------
        # FIX LENGTH — pad / crop to MAX_FRAMES
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
            torch.tensor(label, dtype=torch.float32),
            dataset,
            sample["file_id"]   
        )


# ------------------------------------------------------------
# COLLATE
# ------------------------------------------------------------

def collate_fn(batch):
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]
    d = [b[2] for b in batch]
    f = [b[3] for b in batch]   # file_ids

    return (
        torch.stack(x),
        torch.stack(y),
        d,
        f
    )