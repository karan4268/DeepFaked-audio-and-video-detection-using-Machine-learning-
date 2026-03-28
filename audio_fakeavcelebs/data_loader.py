# =============================================================================
# audio_dataloader.py
# Audio DataLoader for FakeAVCeleb (Binary Audio Classification)
# Video-style pipeline:
#   - per-epoch subsampling
#   - weighted sampling (primary_label)
#   - no static balancing
# =============================================================================

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter, defaultdict


# =============================================================================
# DATASET
# =============================================================================
class AudioDataset(Dataset):
    def __init__(
        self,
        samples,
        root_dir,
        max_time=300,
        augment=False,
        subsample_per_epoch=None
    ):
        self.full_pool = samples
        self.samples = samples
        self.root_dir = root_dir

        self.max_time = max_time
        self.augment = augment
        self.subsample_per_epoch = subsample_per_epoch

    # -------------------------------------------------------------------------
    # PER-EPOCH SUBSAMPLING (same as video) - ensures different samples each epoch to counteract class imbalance and overfitting ovserved duing training.
    # -------------------------------------------------------------------------
    def on_epoch_start(self):
        if self.subsample_per_epoch is None:
            self.samples = self.full_pool
            return

        class_groups = defaultdict(list)
        for s in self.full_pool:
            class_groups[s["primary_label"]].append(s)

        per_class = self.subsample_per_epoch // len(class_groups)

        new_samples = []
        for cls, items in class_groups.items():
            if len(items) >= per_class:
                new_samples.extend(random.sample(items, per_class))
            else:
                new_samples.extend(items)

        random.shuffle(new_samples)
        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------------
    # PAD / CROP
    # -------------------------------------------------------------------------
    def _pad_or_crop(self, x):
        T = x.shape[1]

        if T > self.max_time:
            if self.augment:
                start = random.randint(0, T - self.max_time)
            else:
                start = (T - self.max_time) // 2
            x = x[:, start:start + self.max_time]

        elif T < self.max_time:
            pad_width = self.max_time - T
            x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')

        return x

    # -------------------------------------------------------------------------
    # SPEC AUGMENT
    # -------------------------------------------------------------------------
    def _spec_augment(self, x):
        freq_mask = random.randint(0, 10)
        time_mask = random.randint(0, 20)

        f0 = random.randint(0, max(1, x.shape[0] - freq_mask))
        t0 = random.randint(0, max(1, x.shape[1] - time_mask))

        x[f0:f0 + freq_mask, :] = 0
        x[:, t0:t0 + time_mask] = 0

        return x

    # -------------------------------------------------------------------------
    # GET ITEM
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):

        item = self.samples[idx]
        cache_dir = item["path"]

        npy_path = os.path.join(cache_dir, "logmel.npy")
        label_path = os.path.join(cache_dir, "label.txt")

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing cache: {npy_path}")

        x = np.load(npy_path).astype(np.float32)

        # normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)

        # pad/crop
        x = self._pad_or_crop(x)

        # augment
        if self.augment:
            x = self._spec_augment(x)

        # label
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                primary, binary = f.read().strip().split(",")
                binary = float(binary)
        else:
            binary = float(item["binary_label"])

        y = torch.tensor(binary, dtype=torch.float32)

        x = torch.from_numpy(x).unsqueeze(0)  # (1, 128, T)

        return x, y


# =============================================================================
# LOAD SPLIT
# =============================================================================
def load_split(json_path):

    with open(json_path) as f:
        split = json.load(f)

    train_split = split["train"]
    val_split = split["val"]

    print("\n[DATA DISTRIBUTION]")

    print("Train primary:", Counter([x["primary_label"] for x in train_split]))
    print("Train binary :", Counter([x["binary_label"] for x in train_split]))

    print("Val primary:", Counter([x["primary_label"] for x in val_split]))
    print("Val binary :", Counter([x["binary_label"] for x in val_split]))

    return train_split, val_split


# =============================================================================
# BUILD DATALOADERS
# =============================================================================
def get_audio_dataloaders(
    json_path,
    root_dir,
    batch_size=32,
    num_workers=4,
    subsample_per_epoch=4000
):

    train_samples, val_samples = load_split(json_path)

    print(f"[INFO] Train samples: {len(train_samples)}")
    print(f"[INFO] Val samples: {len(val_samples)}")

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------
    train_dataset = AudioDataset(
        train_samples,
        root_dir,
        augment=True,
        subsample_per_epoch=subsample_per_epoch
    )

    val_dataset = AudioDataset(
        val_samples,
        root_dir,
        augment=False,
        subsample_per_epoch=None
    )

    # initial subsample
    train_dataset.on_epoch_start()

    # -------------------------------------------------------------------------
    # WEIGHTED SAMPLER (PRIMARY LABEL)
    # -------------------------------------------------------------------------
    labels = [s["primary_label"] for s in train_dataset.samples]
    counts = Counter(labels)

    print("[INFO] Class distribution this epoch:", counts)

    total = sum(counts.values())

    class_weights = {
        cls: total / (count + 1e-6)
        for cls, count in counts.items()
    }

    sample_weights = [
        class_weights[s["primary_label"]]
        for s in train_dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # -------------------------------------------------------------------------
    # DATALOADERS
    # -------------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataset, train_loader, val_loader