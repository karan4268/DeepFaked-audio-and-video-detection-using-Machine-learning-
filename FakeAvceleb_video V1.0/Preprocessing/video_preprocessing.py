# ------------------------------------------------------------
# FakeAVCeleb 3D Dataset (JPG frames)
# Class-balanced per-epoch subsampling
# ------------------------------------------------------------

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from collections import defaultdict


class FakeAVCeleb3DDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train_full_pool",
        frames_per_video=24,
        split_json_path=None,
        mode="4class",
        subsample_per_epoch=None,
        transform=None
    ):

        self.root_dir = os.path.abspath(root_dir)
        self.frames_per_video = frames_per_video
        self.mode = mode
        self.subsample_per_epoch = subsample_per_epoch
        self.transform = transform

        if split_json_path is None:
            split_json_path = os.path.join(self.root_dir, "dataset_split_fixed.json")# fixed split with no overlap between train/val/test

        if not os.path.exists(split_json_path):
            raise RuntimeError(f"Split JSON not found at {split_json_path}")

        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        if split not in split_data:
            raise ValueError("split must be train_full_pool / val / test")

        self.full_pool = split_data[split]
        self.samples = self.full_pool.copy()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split: {split}")

        # normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)

        # call once at initialization
        self.on_epoch_start()

    # ------------------------------------------------------------
    # Per Epoch Subsampling
    # ------------------------------------------------------------
    def on_epoch_start(self):

        if self.subsample_per_epoch is None:
            self.samples = self.full_pool.copy()
            return

        if len(self.full_pool) <= self.subsample_per_epoch:
            self.samples = self.full_pool.copy()
            return

        class_to_videos = defaultdict(list)

        for sample in self.full_pool:

            label = sample["primary_label"] if self.mode == "4class" else sample["binary_label"]
            class_to_videos[label].append(sample)

        num_classes = len(class_to_videos)
        per_class = self.subsample_per_epoch // num_classes

        sampled = []

        for label, videos in class_to_videos.items():

            if len(videos) <= per_class:
                sampled.extend(videos)

            else:
                sampled.extend(random.sample(videos, per_class))

        # ensure at least one sample per class
        for label, videos in class_to_videos.items():

            labels_present = [
                s["binary_label"] if self.mode=="video_binary" else s["primary_label"]
                for s in sampled
            ]

            if label not in labels_present:
                sampled.append(random.choice(videos))

        remaining = self.subsample_per_epoch - len(sampled)

        if remaining > 0:
            sampled += random.sample(self.full_pool, remaining)

        random.shuffle(sampled)

        self.samples = sampled

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------
    def _load_clip(self, video_dir):

        # temporal jitter
        max_offset = max(0, 24 - self.frames_per_video)
        start = random.randint(0, max_offset)

        # preallocate tensor (faster than stack)
        clip = torch.empty(3, self.frames_per_video, 224, 224)

        for j, i in enumerate(range(start, start + self.frames_per_video)):

            path = os.path.join(video_dir, f"{i:03d}.jpg")

            if not os.path.isfile(path):
                raise RuntimeError(f"Missing frame: {path}")

            img = Image.open(path).convert("RGB")

            if self.transform:

                img = self.transform(img)

            else:

                img = np.asarray(img, dtype=np.float32) / 255.0
                img = torch.from_numpy(img).permute(2,0,1)

            clip[:, j] = img

        # normalization
        clip = (clip - self.mean) / self.std

        return clip
    
# ------------------------------------------------------------
    def __getitem__(self, idx):

        sample = self.samples[idx]

        video_dir = os.path.normpath(sample["path"])

        clip = self._load_clip(video_dir)

        primary_label = sample["primary_label"]
        binary_label = sample["binary_label"]

        if self.mode == "4class":
            label = primary_label
        else:
            label = binary_label

        return clip, torch.tensor(label, dtype=torch.long), video_dir