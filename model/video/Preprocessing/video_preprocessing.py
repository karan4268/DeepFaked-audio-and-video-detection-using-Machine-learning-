# video_preprocessing_3d.py
# ========================================================= #
# Deepfake Video Dataset (3D CNN - Cache First + Fallback)
# ========================================================= #
#
# CHANGES FROM ORIGINAL:
#   [FIX 1] _load_cached() now applies temporal jitter for the train split.
#           Original always loaded all 24 PNGs in sorted order, making the
#           jitter logic in sample_frame_ids() dead code for cached videos.
#           Now train randomly picks a contiguous window of frames_per_video
#           from the 24 cached PNGs each epoch.
#
#   [FIX 2] Added make_sampler() — a WeightedRandomSampler factory that
#           balances real/fake classes.  FF++ has ~5:1 fake:real; without
#           balancing the model predicts "fake" for everything and reports
#           a deceptively high AUC from epoch 1.
#           Usage: pass the returned sampler to DataLoader (see docstring).
#
#   Everything else is unchanged.
# ========================================================= #

import os
import cv2
import torch
import random
import hashlib
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


# ========================================================= #
# DATASET
# ========================================================= #

class DeepfakedDataset(Dataset):
    """
    Output:
        frames : FloatTensor [C, T, 224, 224]
        label  : LongTensor  (0 = real, 1 = fake)

    Typical usage
    -------------
        train_ds = DeepfakedDataset(root, split="train")
        sampler  = make_sampler(train_ds)           # [FIX 2]
        loader   = DataLoader(train_ds, batch_size=16, sampler=sampler)

        val_ds   = DeepfakedDataset(root, split="val")
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    """

    def __init__(
        self,
        root_dir,
        split="train",
        frames_per_video=24,
        use_fallback=True,
        seed=42,
        device=None
    ):
        self.root_dir         = os.path.abspath(root_dir)
        self.split            = split
        self.frames_per_video = frames_per_video
        self.use_fallback     = use_fallback
        self.device           = device or ("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(seed)
        np.random.seed(seed)

        # --------------------------------------------------
        # Cache root
        # --------------------------------------------------
        self.cache_root = os.path.join(self.root_dir, "cached_faces", split)

        # --------------------------------------------------
        # Dataset paths
        # --------------------------------------------------
        split_dir = os.path.join(self.root_dir, split)
        real_dir  = os.path.join(split_dir, "real")
        fake_dir  = os.path.join(split_dir, "fake")

        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            raise FileNotFoundError("Real/Fake folders missing.")

        self.samples = []

        for f in sorted(os.listdir(real_dir)):
            if f.endswith(".mp4"):
                self.samples.append((os.path.join(real_dir, f), 0))

        for root, _, files in os.walk(fake_dir):
            for f in sorted(files):
                if f.endswith(".mp4"):
                    self.samples.append((os.path.join(root, f), 1))

        if not self.samples:
            raise RuntimeError("No videos found.")

        random.shuffle(self.samples)

        # labels exposed for make_sampler()
        self.labels = [label for _, label in self.samples]

        # --------------------------------------------------
        # Transforms (train vs eval)
        # --------------------------------------------------
        if self.split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # --------------------------------------------------
        # Face detector (fallback only)
        # --------------------------------------------------
        self.face_detector = None
        if use_fallback:
            if MTCNN is None:
                raise ImportError("facenet-pytorch required for fallback.")
            self.face_detector = MTCNN(
                image_size=224,
                margin=20,
                keep_all=False,
                device="cpu"
            )

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def _video_cache_dir(self, video_path):
        rel = os.path.relpath(
            video_path,
            os.path.join(self.root_dir, self.split)
        )
        h = hashlib.md5(rel.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_root, h)

    # --------------------------------------------------
    # Uniform temporal sampling (fallback path only)
    # --------------------------------------------------
    def sample_frame_ids(self, total_frames):
        if self.split == "train":
            if total_frames <= self.frames_per_video:
                return (
                    list(range(total_frames))
                    + [total_frames - 1] * (self.frames_per_video - total_frames)
                )
            start = random.randint(0, total_frames - self.frames_per_video)
            return list(range(start, start + self.frames_per_video))
        else:
            return np.linspace(
                0, total_frames - 1, self.frames_per_video, dtype=int
            ).tolist()

    # --------------------------------------------------
    # Load from cache
    # --------------------------------------------------
    def _load_cached(self, cache_dir):
        all_files = sorted(
            [f for f in os.listdir(cache_dir) if f.endswith(".png")],
            key=lambda x: int(x.split(".")[0])
        )

        if len(all_files) < self.frames_per_video:
            raise RuntimeError(f"Incomplete cache: {cache_dir}")

        # ------------------------------------------------
        # [FIX 1] Temporal jitter for train split.
        # Original always used all 24 files in order — jitter was dead code
        # for cached videos.  Now we pick a random contiguous window so the
        # model sees a different clip start each epoch, matching the intent
        # of sample_frame_ids() in the fallback path.
        # Val/test remain fully deterministic (all frames, sorted order).
        # ------------------------------------------------
        if self.split == "train" and len(all_files) > self.frames_per_video:
            start      = random.randint(0, len(all_files) - self.frames_per_video)
            file_slice = all_files[start : start + self.frames_per_video]
        else:
            file_slice = all_files[:self.frames_per_video]

        frames = []
        for fname in file_slice:
            path = os.path.join(cache_dir, fname)
            with Image.open(path) as img:
                img = img.convert("RGB")
                frames.append(self.transform(img))

        return torch.stack(frames)

    # --------------------------------------------------
    # Fallback decoding (cache miss)
    # --------------------------------------------------
    def _decode_video(self, video_path):
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros(self.frames_per_video, 3, 224, 224)

        base_ids = self.sample_frame_ids(total_frames)

        if self.split == "train":
            jitter    = np.random.randint(-2, 3, size=len(base_ids))
            frame_ids = np.clip(np.array(base_ids) + jitter, 0, total_frames - 1)
        else:
            frame_ids = base_ids

        frames = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.face_detector is not None:
                boxes, probs = self.face_detector.detect(frame)
                if boxes is not None and probs is not None and probs[0] > 0.9:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    frame = frame[y1:y2, x1:x2]

            frame = Image.fromarray(frame).resize((224, 224))
            frames.append(self.transform(frame))

        cap.release()

        if not frames:
            return torch.zeros(self.frames_per_video, 3, 224, 224)

        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        return torch.stack(frames[:self.frames_per_video])

    # --------------------------------------------------
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cache_dir         = self._video_cache_dir(video_path)

        if os.path.isdir(cache_dir):
            frames = self._load_cached(cache_dir)
        else:
            if not self.use_fallback:
                raise RuntimeError(f"Cache missing: {video_path}")
            frames = self._decode_video(video_path)

        # [T, C, H, W] → [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3)

        return frames, torch.tensor(label, dtype=torch.long)


# ========================================================= #
# [FIX 2] WEIGHTED SAMPLER FACTORY
# ========================================================= #

def make_sampler(dataset: DeepfakedDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives each class equal expected
    frequency per batch, regardless of dataset imbalance.

    FF++ has roughly 5:1 fake:real.  Without this, the model predicts
    "fake" for everything and reports a deceptively high AUC from epoch 1.

    Usage:
        sampler = make_sampler(train_dataset)
        loader  = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        # Do NOT pass shuffle=True when using a sampler.
    """
    labels        = np.array(dataset.labels)
    class_counts  = np.bincount(labels)                  # [n_real, n_fake]
    class_weights = 1.0 / class_counts                   # inverse frequency
    sample_weights = class_weights[labels]               # per-sample weight

    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True
    )
