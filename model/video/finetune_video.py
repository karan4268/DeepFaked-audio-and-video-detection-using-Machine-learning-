# =============================================================================
# finetune_video.py
#
# Fine-tunes the existing R3D-18 VideoModel checkpoint on FakeAVCeleb.
# Reads individual frames.npy files from cache.
#
# Usage:
#   python finetune_video.py                        # fresh from base checkpoint
#   python finetune_video.py --freeze_backbone      # head-only warmup
#   python finetune_video.py \
#       --resume ".../finetuned_best.pth"           # resume interrupted run
# =============================================================================

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from video_model import VideoModel

# =============================================================================
# PATHS
# =============================================================================

CHECKPOINT_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_combined\combined_train_best.pth"
)
CACHE_ROOT = r"D:\FakeAVCache\Video"
OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_finetuned"
)
SPLITS_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

NUM_FRAMES         = 24
SAMPLES_PER_EPOCH  = 2000
BATCH_SIZE         = 4
HEAD_LR            = 1e-4
BACKBONE_LR        = 1e-5
WEIGHT_DECAY       = 1e-4
EPOCHS             = 30
PATIENCE           = 10
WARMUP_EPOCHS      = 2
NUM_WORKERS        = 4


# =============================================================================
# DATASET
# =============================================================================

class NpyDataset(Dataset):

    def __init__(self, samples, cache_root, augment=False, num_frames=24):
        self.samples    = samples
        self.cache_root = cache_root
        self.augment    = augment
        self.num_frames = num_frames
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        # meta["file"] is cache-root-relative e.g. "abc123/frames.npy"
        path = os.path.join(self.cache_root, meta["file"])
        clip = np.load(path)                               # (T, H, W, C) uint8

        T    = clip.shape[0]
        idxs = np.linspace(0, T - 1, self.num_frames).astype(int)
        clip = clip[idxs].astype(np.float32) / 255.0

        clip = np.transpose(clip, (0, 3, 1, 2))           # (T, C, H, W)
        clip = (clip - self.mean) / self.std

        if self.augment and np.random.rand() < 0.5:
            clip = clip[:, :, :, ::-1].copy()             # random horizontal flip

        clip = torch.from_numpy(clip).permute(1, 0, 2, 3) # (C, T, H, W)
        return clip, torch.tensor(meta["label"], dtype=torch.long)


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch])
    )


# =============================================================================
# SAMPLER UTILS
# =============================================================================

def make_sampler(labels_list):
    labels  = np.array(labels_list)
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True
    )


def subsample_balanced(labels_arr, n_total, seed=None):
    rng      = np.random.default_rng(seed)
    real_idx = np.where(labels_arr == 0)[0]
    fake_idx = np.where(labels_arr == 1)[0]
    n_each   = n_total // 2
    chosen   = np.concatenate([
        rng.choice(real_idx, min(n_each, len(real_idx)), replace=False),
        rng.choice(fake_idx, min(n_each, len(fake_idx)), replace=False),
    ])
    return chosen


def build_train_loader(train_ds, labels_arr, batch_size, num_workers,
                       samples_per_epoch, epoch_seed):
    chosen  = subsample_balanced(labels_arr, samples_per_epoch, seed=epoch_seed)
    subset  = Subset(train_ds, chosen.tolist())
    sampler = make_sampler(labels_arr[chosen].tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        collate_fn=collate_fn,
    )


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def compute_auc(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  Val", ncols=90, leave=False):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
    return roc_auc_score(all_labels, all_scores)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="  Train", ncols=90, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(n, 1)


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    val_auc, train_loss, patience_ctr, num_frames):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "val_auc":              val_auc,
        "train_loss":           train_loss,
        "patience_ctr":         patience_ctr,
        "num_frames":           num_frames,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """
    Restores full training state from a checkpoint.
    Returns (start_epoch, best_val_auc, patience_ctr).
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch  = ckpt.get("epoch", 0)
    best_val_auc = ckpt.get("val_auc", 0.0)
    patience_ctr = ckpt.get("patience_ctr", 0)
    saved_frames = ckpt.get("num_frames", None)

    return start_epoch, best_val_auc, patience_ctr, saved_frames


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        default=CHECKPOINT_PATH,
                        help="Base checkpoint to start from (ignored if --resume set)")
    parser.add_argument("--cache_root",        default=CACHE_ROOT)
    parser.add_argument("--output_dir",        default=OUTPUT_DIR)
    parser.add_argument("--splits_path",       default=SPLITS_PATH)
    parser.add_argument("--batch_size",        type=int,   default=BATCH_SIZE)
    parser.add_argument("--head_lr",           type=float, default=HEAD_LR)
    parser.add_argument("--backbone_lr",       type=float, default=BACKBONE_LR)
    parser.add_argument("--epochs",            type=int,   default=EPOCHS)
    parser.add_argument("--patience",          type=int,   default=PATIENCE)
    parser.add_argument("--num_frames",        type=int,   default=NUM_FRAMES)
    parser.add_argument("--samples_per_epoch", type=int,   default=SAMPLES_PER_EPOCH)
    parser.add_argument("--num_workers",       type=int,   default=NUM_WORKERS)
    parser.add_argument("--freeze_backbone",   action="store_true")
    parser.add_argument("--resume",            default=r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video_finetuned\finetuned_best.pth",
                        help="Path to interrupted finetuned checkpoint to resume from. "
                             "Restores model, optimizer, scheduler, scaler, epoch, "
                             "patience counter. Omit to start fresh.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device          : {device} | {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Num frames      : {args.num_frames}")
    print(f"[INFO] Samples / epoch : {args.samples_per_epoch}")

    # -------------------------------------------------------------------------
    # Index + splits
    # -------------------------------------------------------------------------
    print("\n[INFO] Loading video index...")
    with open(os.path.join(args.cache_root, "video_index.json")) as f:
        index = json.load(f)
    print(f"  Total : {len(index)} samples")

    print("[INFO] Loading splits...")
    with open(args.splits_path) as f:
        splits = json.load(f)

    train_idx     = np.array(splits["train_indices"])
    val_idx       = np.array(splits["val_indices"])
    train_samples = [index[i] for i in train_idx]
    val_samples   = [index[i] for i in val_idx]

    t_real = sum(1 for s in train_samples if s["label"] == 0)
    t_fake = sum(1 for s in train_samples if s["label"] == 1)
    v_real = sum(1 for s in val_samples   if s["label"] == 0)
    v_fake = sum(1 for s in val_samples   if s["label"] == 1)

    print(f"  Train : {len(train_samples):5d}  ({t_real} real, {t_fake} fake)  "
          f"{len(splits['train_speakers'])} speakers")
    print(f"  Val   : {len(val_samples):5d}  ({v_real} real, {v_fake} fake)  "
          f"{len(splits['val_speakers'])} speakers")
    print(f"  Test  : held out — {len(splits['test_speakers'])} speakers, never touched")

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    train_ds   = NpyDataset(train_samples, args.cache_root,
                            augment=True,  num_frames=args.num_frames)
    val_ds     = NpyDataset(val_samples,   args.cache_root,
                            augment=False, num_frames=args.num_frames)
    labels_arr = np.array([s["label"] for s in train_samples])

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------------------
    # Model — always load base checkpoint first, then optionally overwrite
    # with resume checkpoint below (after optimizer is built)
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Loading base checkpoint: {args.checkpoint}")
    model = VideoModel(num_classes=2, pretrained=False, dropout=0.3).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.freeze_backbone:
        print("  [MODE] Backbone FROZEN — head only")
        for name, param in model.named_parameters():
            if "backbone.fc" not in name:
                param.requires_grad = False
    else:
        print("  [MODE] Full fine-tuning (differential LR)")

    # -------------------------------------------------------------------------
    # Optimizer / scheduler / scaler / loss
    # Must be built BEFORE resume so load_state_dict has objects to populate.
    # -------------------------------------------------------------------------
    head_params     = list(model.backbone.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone.fc" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.head_lr},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        p = (epoch - WARMUP_EPOCHS) / max(args.epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler()

    # Sampler already produces balanced 50/50 batches every epoch.
    # Using class weights on top of a balanced sampler is double correction
    # and inflates the loss above random-chance baseline (~0.69).
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # RESUME — restores all training state on top of freshly built objects
    # -------------------------------------------------------------------------
    start_epoch  = 0
    best_val_auc = 0.0
    patience_ctr = 0

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        start_epoch, best_val_auc, patience_ctr, saved_frames = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

        # Safety check — warn if frame count differs from saved checkpoint
        if saved_frames is not None and saved_frames != args.num_frames:
            print(f"\n  [WARNING] Checkpoint was saved with num_frames={saved_frames} "
                  f"but current --num_frames={args.num_frames}.")
            print(f"  [WARNING] Resuming with mismatched frame count. "
                  f"Pass --num_frames {saved_frames} to match the original run.\n")

        print(f"  Resumed : epoch={start_epoch}  "
              f"best_val_auc={best_val_auc:.4f}  "
              f"patience={patience_ctr}/{args.patience}")
    else:
        print("\n[INFO] Starting fresh from base checkpoint")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    history = []

    print(f"\n[INFO] Starting — epochs {start_epoch+1}→{args.epochs}, "
          f"patience={args.patience}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Fresh balanced subsample every epoch — deterministic per epoch number
        # so resuming reproduces the same sequence of subsamples
        train_loader = build_train_loader(
            train_ds, labels_arr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            samples_per_epoch=args.samples_per_epoch,
            epoch_seed=epoch,          # same seed → same subsample if restarted
        )

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler, device)
        val_auc    = compute_auc(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
              f"lr_h={optimizer.param_groups[1]['lr']:.1e}  "
              f"lr_bb={optimizer.param_groups[0]['lr']:.1e}  "
              f"time={elapsed/60:.1f}min")

        history.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_auc":    val_auc,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            save_checkpoint(
                path         = os.path.join(args.output_dir, "finetuned_best.pth"),
                epoch        = epoch + 1,
                model        = model,
                optimizer    = optimizer,
                scheduler    = scheduler,
                scaler       = scaler,
                val_auc      = val_auc,
                train_loss   = train_loss,
                patience_ctr = patience_ctr,
                num_frames   = args.num_frames,
            )
            print(f"  ✓ New best  (val_auc={best_val_auc:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{args.patience})")
            if patience_ctr >= args.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch + 1}")
                break

        # Periodic checkpoint every 5 epochs — also fully resume-friendly
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                path         = os.path.join(args.output_dir,
                                            f"finetuned_epoch{epoch+1}.pth"),
                epoch        = epoch + 1,
                model        = model,
                optimizer    = optimizer,
                scheduler    = scheduler,
                scaler       = scaler,
                val_auc      = val_auc,
                train_loss   = train_loss,
                patience_ctr = patience_ctr,
                num_frames   = args.num_frames,
            )

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------
    with open(os.path.join(args.output_dir, "finetune_history.json"), "w") as f:
        json.dump({"best_val_auc": best_val_auc, "history": history}, f, indent=2)

    print(f"\n[INFO] Done.  Best val AUC : {best_val_auc:.4f}")
    print(f"[INFO] Saved → {os.path.join(args.output_dir, 'finetuned_best.pth')}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()