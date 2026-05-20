# =============================================================================
# val_video_on_split.py
# Runs video model inference on the VAL split of FakeAVCeleb.
# Outputs val scores + file_ids used for oracle-free alpha selection in fusion.
# =============================================================================

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from video_model import VideoModel

# =============================================================================
# PATHS
# =============================================================================

MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_finetuned\finetuned_best.pth"
)

CACHE_ROOT = r"D:\FakeAVCache\Video"

SPLITS_JSON = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)

OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_finetuned\eval_val_for_alpha"
)

NUM_FRAMES = 24


# =============================================================================
# DATASET  (identical to test eval — deterministic, no augmentation)
# =============================================================================

class ValSplitDataset(Dataset):

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

        # meta["file"] is cache-root-relative e.g. "abc123/frames.npy"
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
        clip = np.transpose(clip, (0, 3, 1, 2))           # (T, C, H, W)
        clip = (clip - self.mean) / self.std
        clip = torch.from_numpy(clip).permute(1, 0, 2, 3) # (C, T, H, W)

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devicename = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    use_amp = torch.cuda.is_available()
    print(f"[INFO] Device: {device} | {devicename}")

    # -------------------------------------------------------------------------
    # Load splits + index
    # -------------------------------------------------------------------------
    print("[INFO] Loading splits...")
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    print("[INFO] Loading video index...")
    with open(os.path.join(CACHE_ROOT, "video_index.json")) as f:
        index = json.load(f)

    val_indices = splits["val_indices"]
    val_samples = [index[i] for i in val_indices]

    real_count = sum(1 for s in val_samples if s["label"] == 0)
    fake_count = sum(1 for s in val_samples if s["label"] == 1)

    print(f"  Val samples : {len(val_samples)}")
    print(f"  Real        : {real_count}")
    print(f"  Fake        : {fake_count}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Loading model: {MODEL_PATH}")
    model = VideoModel(num_classes=2, pretrained=False, dropout=0.3).to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Checkpoint val_auc : {ckpt.get('val_auc', '?')}")

    # -------------------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------------------
    dataset = ValSplitDataset(val_samples, CACHE_ROOT, NUM_FRAMES)

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_labels  = []
    all_scores  = []
    all_indices = []

    with torch.no_grad():
        for x, y, ids in tqdm(loader, desc="Video Val Inference"):
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

    all_labels   = np.array(all_labels)
    all_scores   = np.array(all_scores)
    all_indices  = np.array(all_indices)
    all_file_ids = np.array([val_samples[i]["file_id"] for i in all_indices])

    print(f"\n[INFO] Done — {len(all_labels)} samples")
    print(f"[INFO] Val AUC (video): {roc_auc_score(all_labels, all_scores):.4f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "video_val_scores.npy"),   all_scores)
    np.save(os.path.join(OUTPUT_DIR, "video_val_labels.npy"),   all_labels)
    np.save(os.path.join(OUTPUT_DIR, "video_val_file_ids.npy"), all_file_ids)

    print(f"[INFO] Saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()