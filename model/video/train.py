# train_combined.py
# Train video model on FF++ + CelebDF combined, using cached faces for both.
# Sampler balances real/fake across the merged dataset.
# Output model used as video encoder in fusion stage (FakeAVCelebs).
#
# DATASET ROOTS:
#   FF++    : Data/Video/           (train/real, train/fake structure)
#   CelebDF : Data/Video/CelebDF/   (train/real, train/fake structure)
#
# BOTH must have cached_faces/ already built via cache_face.py
#
# WHY COMBINED:
#   FF++    — large, compression artifacts, good generalization base
#   CelebDF — high quality renders, identity-based artifacts, closer to FakeAVCelebs
#   Together they cover both artifact types the fusion stage will encounter
# =============================================================================

import os
import sys
import time
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from Preprocessing.video_preprocessing import DeepfakedDataset
from video_model import VideoModel
from eval_video import evaluate


# =============================================================================
# CONFIG
# =============================================================================

FFPP_ROOT    = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23"
CELEBDF_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF"
MODEL_DIR    = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video_combined"

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "combined_train.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "combined_train_best.pth")
LOG_PATH        = os.path.join(MODEL_DIR, "combined_train_log.csv")

FRAMES_PER_VIDEO   = 24     #24 Frames per video (≈1s at 24fps)
BATCH_SIZE         = 3      # safe max for GTX 1650 Ti + R3D-18 + 24 frames
ACCUMULATION_STEPS = 4      # effective batch size = 12
NUM_WORKERS        = 0      # number of CPU threads to use for data loading (my case 0 is better)
PREFETCH_FACTOR    = 2      # prefetch batches in background using threading instead
NUM_EPOCHS         = 15     # number of epochs to train for 
LR                 = 1e-4   # learning rate
EARLY_STOP_PAT     = 5      # early stopping patience (number of epochs with no AUC improvement before stopping)

# ------------------------------------------------------------------
# SUBSAMPLE CONFIG
# Trains on a stratified subset each run — faster iteration without
# losing coverage of both domains.
# 2000 train (~4h/epoch) is a good balance of speed vs learning signal.
# Set TRAIN_SAMPLES = None to use the full 8843-sample dataset.
# ------------------------------------------------------------------
TRAIN_SAMPLES = 2000   # ~286 real + ~1714 fake, both FF++ and CelebDF combined
VAL_SAMPLES   = None    # full Val set

# kept for quick smoke-test only — use TRAIN_SAMPLES for real runs
DEBUG_MODE          = False
DEBUG_TRAIN_SAMPLES = 400
DEBUG_VAL_SAMPLES   = 200


# =============================================================================
# STRATIFIED SUBSET
# =============================================================================

def stratified_subset(dataset, n, seed=None):
    """
    Stratified subset preserving real/fake ratio.
    seed=None  → random each call (use for train — different videos each epoch)
    seed=int   → deterministic (use for val — same samples every epoch)
    """
    labels  = np.array(dataset.labels)
    idx     = list(range(len(dataset)))
    n       = min(n, len(dataset))
    keep, _ = train_test_split(idx, train_size=n, stratify=labels, random_state=seed)
    sub        = Subset(dataset, keep)
    sub.labels = [dataset.labels[i] for i in keep]
    return sub


# =============================================================================
# CONCAT DATASET WITH LABELS
# Wraps ConcatDataset and exposes a flat .labels list for the sampler
# =============================================================================

class LabelledConcatDataset(ConcatDataset):
    """ConcatDataset that also exposes a flat .labels list."""
    def __init__(self, datasets):
        super().__init__(datasets)
        self.labels = []
        for ds in datasets:
            self.labels.extend(ds.labels)


# =============================================================================
# WEIGHTED SAMPLER
# =============================================================================

def make_sampler(dataset):
    labels        = np.array(dataset.labels)
    class_counts  = np.bincount(labels)
    sample_weights = (1.0 / class_counts)[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True
    )


# =============================================================================
# PROGRESS BAR
# =============================================================================

def _bar(done, total, width=20):
    filled = int(width * done / total)
    arrow  = ">" if filled < width else ""
    spaces = " " * (width - filled - len(arrow))
    return f"[{'=' * filled}{arrow}{spaces}]"

def print_progress(batch, total_batches, loss, elapsed, epoch, num_epochs):
    bar      = _bar(batch, total_batches)
    avg_t    = elapsed / max(batch, 1)
    eta_secs = avg_t * (total_batches - batch)
    if eta_secs >= 3600:
        eta = f"{int(eta_secs//3600)}h {int((eta_secs%3600)//60)}m"
    elif eta_secs >= 60:
        eta = f"{int(eta_secs//60)}m {int(eta_secs%60)}s"
    else:
        eta = f"{int(eta_secs)}s"
    sys.stdout.write(
        f"\r  Epoch {epoch}/{num_epochs}  {bar}  "
        f"{batch:>4}/{total_batches}  "
        f"loss {loss:.4f}  "
        f"{avg_t:.2f}s/b  "
        f"ETA {eta:<10}"
    )
    sys.stdout.flush()


# =============================================================================
# TIME FORMATTING for logs and progress display
# =============================================================================

def format_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device name: {device} | {device_name}")
    if DEBUG_MODE:
        print(f"[DEBUG] Debug mode ON — "
              f"train capped at {DEBUG_TRAIN_SAMPLES}, val at {DEBUG_VAL_SAMPLES}")

    # ------------------------------------------------------------------
    # Load both datasets separately then concatinate
    # Val uses CelebDF only — it's the harder domain and closer to FakeAVCelebs
    # ------------------------------------------------------------------
    print("\n[INFO] Loading datasets...")

    ffpp_train    = DeepfakedDataset(
        root_dir=FFPP_ROOT, split="train",
        frames_per_video=FRAMES_PER_VIDEO, device=device.type
    )
    celebdf_train = DeepfakedDataset(
        root_dir=CELEBDF_ROOT, split="train",
        frames_per_video=FRAMES_PER_VIDEO, device=device.type
    )

    # Val: CelebDF only (harder domain, closer to FakeAVCelebs test set)
    val_dataset = DeepfakedDataset(
        root_dir=CELEBDF_ROOT, split="val",
        frames_per_video=FRAMES_PER_VIDEO, device=device.type
    )

    # Stats before concatatination
    ffpp_real    = ffpp_train.labels.count(0)
    ffpp_fake    = ffpp_train.labels.count(1)
    celeb_real   = celebdf_train.labels.count(0)
    celeb_fake   = celebdf_train.labels.count(1)

    print(f"[INFO] FF++    train: {len(ffpp_train):>5} samples  "
          f"(real: {ffpp_real}, fake: {ffpp_fake})")
    print(f"[INFO] CelebDF train: {len(celebdf_train):>5} samples  "
          f"(real: {celeb_real}, fake: {celeb_fake})")

    # Concatatinate datasets and labels for unified sampling
    train_dataset = LabelledConcatDataset([ffpp_train, celebdf_train])

    total_real  = train_dataset.labels.count(0)
    total_fake  = train_dataset.labels.count(1)
    print(f"[INFO] Combined train: {len(train_dataset):>5} samples  "
          f"(real: {total_real}, fake: {total_fake}, "
          f"ratio: {total_fake/total_real:.2f})")
    print(f"[INFO] Val (CelebDF): {len(val_dataset):>5} samples  "
          f"(real: {val_dataset.labels.count(0)}, "
          f"fake: {val_dataset.labels.count(1)})")

    # Val subsample — full set
    # Static val is essential — AUC changes mean model improvement, not sample luck.
    if DEBUG_MODE:
        val_dataset = stratified_subset(val_dataset, DEBUG_VAL_SAMPLES, seed=42)
    elif VAL_SAMPLES is not None: # if val sample is not none, then subsample it
        val_dataset = stratified_subset(val_dataset, VAL_SAMPLES, seed=42)

    vl = val_dataset.labels
    print(f"[INFO] Val: {len(val_dataset)} samples  "
        f"(real: {vl.count(0)}, fake: {vl.count(1)})  "
        f"{'full set' if VAL_SAMPLES is None else f'subsampled from {len(val_dataset)}'}")
    print(f"[INFO] Train pool: {len(train_dataset)} samples — "
          f"resampling {DEBUG_TRAIN_SAMPLES if DEBUG_MODE else TRAIN_SAMPLES} each epoch")

    # ------------------------------------------------------------------
    # Val loader — fixed, uses full val set or a fixed subsample 
    # Train loader — rebuilt each epoch with a fresh random subsample
    #   so the model sees different videos every epoch while staying fast
    # ------------------------------------------------------------------
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=False
    )
    # train_loader is built inside the epoch loop (see below)
    n_train_samples = DEBUG_TRAIN_SAMPLES if DEBUG_MODE else TRAIN_SAMPLES

    # ------------------------------------------------------------------
    # Model and training setup
    # ------------------------------------------------------------------
    model = VideoModel().to(device)
    print(f"\n[INFO] Training on FF++ + CelebDF combined")

    # Sampler already balances — no class_weight in loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # Resume checkpoint setup
    # ------------------------------------------------------------------
    start_epoch   = 0
    best_auc      = 0.0
    epochs_no_imp = 0

    if os.path.exists(CHECKPOINT_PATH):
        print("[INFO] Loading checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_auc      = ckpt.get("best_auc", 0.0)
        epochs_no_imp = ckpt.get("epochs_no_imp", 0)
        if "scheduler_state_dict" in ckpt:                          
            scheduler.load_state_dict(ckpt["scheduler_state_dict"]) 
        print(f"[INFO] Resumed from epoch {start_epoch}  |  Best AUC: {best_auc:.4f}")

    # ------------------------------------------------------------------
    # CSV logging setup
    # ------------------------------------------------------------------
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_accuracy",
                "val_auc", "is_best", "lr", "timestamp"
            ])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'[DEBUG] ' if DEBUG_MODE else ''}Training up to {NUM_EPOCHS} epochs  |  "
          f"Patience: {EARLY_STOP_PAT}  |  "
          f"Effective batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print("-" * 60)

    training_start = time.time()
    epoch_times    = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        current_lr  = scheduler.get_last_lr()[0]
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]  LR={current_lr:.2e}")

        # Fresh stratified subsample every epoch — seed=None means
        # different 2000 videos each epoch from the full 8843 pool
        epoch_train = stratified_subset(train_dataset, n_train_samples, seed=None)
        sampler     = make_sampler(epoch_train)
        train_loader = DataLoader(
            epoch_train, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False
        )
        total_batches = len(train_loader)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        batch_start  = time.time()

        for i, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss = criterion(model(frames), labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUMULATION_STEPS
            print_progress(i + 1, total_batches, running_loss / (i + 1),
                           time.time() - batch_start, epoch + 1, NUM_EPOCHS)

        # flush leftover accumulation
        if total_batches % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss      = running_loss / max(total_batches, 1)
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        eta_total     = (sum(epoch_times) / len(epoch_times)) * (NUM_EPOCHS - epoch - 1)

        print(f"\n  ↳ loss {avg_loss:.4f}  |  "
              f"took {format_duration(epoch_elapsed)}  |  "
              f"elapsed {format_duration(time.time() - training_start)}  |  "
              f"ETA {format_duration(eta_total)}")

        # validation after each epoch
        model.eval()
        val_acc, val_auc = evaluate(model, val_loader, device, tta=1)
        print(f"  ↳ val  acc {val_acc:.4f}  auc {val_auc:.4f}")

        scheduler.step()

        # save best model based on AUC + checkpoint every epoch for resume functionality
        is_best = val_auc > best_auc + 1e-4
        if is_best:
            best_auc      = val_auc
            epochs_no_imp = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc, "epochs_no_imp": epochs_no_imp,
                "scheduler_state_dict": scheduler.state_dict()   # ← add
            }, BEST_MODEL_PATH)
            print(f"  ✨ Best model saved  (AUC={best_auc:.4f})")
        else:
            epochs_no_imp += 1
            print(f"  No improvement — patience {epochs_no_imp}/{EARLY_STOP_PAT}")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc, "epochs_no_imp": epochs_no_imp,
            "scheduler_state_dict": scheduler.state_dict() 
        }, CHECKPOINT_PATH)
        print("  🔶 Checkpoint saved")

        with open(LOG_PATH, "a", newline="") as f:  # save csv log after every epoch
            csv.writer(f).writerow([
                epoch + 1, round(avg_loss, 6), round(val_acc, 6),
                round(val_auc, 6), int(is_best),
                f"{current_lr:.2e}", time.strftime("%Y-%m-%d %H:%M:%S")
            ])

        if epochs_no_imp >= EARLY_STOP_PAT:
            print(f"\n[INFO] Early stopping — no improvement for {epochs_no_imp} epochs.")
            break

    print(f"\n{'[DEBUG] ' if DEBUG_MODE else ''}Done.  Best val AUC: {best_auc:.4f}")
    print(f"   Total time: {format_duration(time.time() - training_start)}")
    print(f"   Best model: {BEST_MODEL_PATH}")
    if DEBUG_MODE:
        print("\n  -> Set DEBUG_MODE <- to  *False* and rerun for full training.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()