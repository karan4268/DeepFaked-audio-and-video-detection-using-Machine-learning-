# =============================================================================
# train.py
# Training script for Deepfake Video Detection using VideoModel (ResNet18 3D)
# added Gradient Accumulation and Class-Balanced Subsampling per epoch to handle class imbalance and memory constraints.
# Each epoch samples a new subset of videos from the full training pool.
# Supports per-epoch class-balanced subsampling and binary weighting
# implemented a subsampling strategy that ensures at least 2 classes are present in each epoch, and applies class weights to the loss function to handle imbalance.
# The training loop includes checkpointing, best model saving, and CSV logging of metrics and class distribution.
# each epoch gets a new random subset of videos from the full pool, ensuring at least 2 classes are present
# Train: 1000 random videos from full pool
# Val: full validation pool
# Logs class distribution per epoch to ensure 2-class minimum
# Primary labels (4-class):
#   0: RealVideo-RealAudio (RR)
#   1: FakeVideo-RealAudio (FR)
#   2: RealVideo-FakeAudio (RF)
#   3: FakeVideo-FakeAudio (FF)
# ------------------------------------------------------------
# Binary labels (for video-only model):
#   0: Real (RR + RF)
#   1: Fake (FR + FF)
# ------------------------------------------------------------
# VIDEO_BINARY_MAP =
#    0: 0  # RR -> Real
#    1: 1  # FR -> Fake
#    2: 0  # RF -> Real
#    3: 1  # FF -> Fake
# ------------------------------------------------------------
# =============================================================================

import os
import time
import csv
import torch
import torch.nn as nn
import multiprocessing
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

from prefetch_loader import CUDAPrefetcher  # cuda prefetcher to overlap data loading with GPU computation
from Preprocessing.video_preprocessing import FakeAVCeleb3DDataset
from video_model import VideoModel
from eval_video import evaluate


# =============================================================================
# Config
# =============================================================================

ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"

CHECKPOINT_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\video_model.pth_3D_CNN-ResNet18"
BEST_MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\video_model_best_3D_CNN-ResNet18.pth"
LOG_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\training_log_3D_CNN-ResNet18.csv"

SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "dataset_split.json")

FRAMES_PER_VIDEO = 24       # number of frames sampled from each video in cache and during training/validation.
BATCH_SIZE = 2              # effective batch size will be BATCH_SIZE * ACCUMULATION_STEPS. 2 is fine for my GPU
ACCUMULATION_STEPS = 8      # number of steps to accumulate gradients before updating model weights (to simulate larger batch size and reduce memory usage)

NUM_WORKERS = min(4, os.cpu_count())
NUM_EPOCHS = 15

SUBSAMPLE_PER_EPOCH = 1000  # number of videos to sample from the full training pool at the start of each epoch (set to None to disable subsampling and use full pool every epoch)

MODE = "video_binary"


# =============================================================================
# Training
# =============================================================================

def main():

    # =============================================================================
    # Device Setup
    # =============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"[INFO] Using device: {device} | AMP: {use_amp}")

    # =============================================================================
    # Dataset
    # =============================================================================

    train_dataset = FakeAVCeleb3DDataset(
        root_dir=ROOT_DIR,
        split="train_full_pool",
        frames_per_video=FRAMES_PER_VIDEO,
        split_json_path=SPLIT_JSON_PATH,
        mode=MODE,
        subsample_per_epoch=SUBSAMPLE_PER_EPOCH
    )

    val_dataset = FakeAVCeleb3DDataset(
        root_dir=ROOT_DIR,
        split="val",
        frames_per_video=FRAMES_PER_VIDEO,
        split_json_path=SPLIT_JSON_PATH,
        mode=MODE,
    )

    print(f"[INFO] Train samples (full pool): {len(train_dataset.full_pool)}")
    print(f"[INFO] Val samples (full pool): {len(val_dataset.full_pool)}")

    # =============================================================================
    # Model
    # =============================================================================

    model = VideoModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # =============================================================================
    # Resume Training
    # =============================================================================

    start_epoch = 0
    best_auc = 0.0

    if os.path.exists(CHECKPOINT_PATH):

        print("[INFO] Loading checkpoint...")

        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)

        print(f"[INFO] Resumed from epoch {start_epoch} | Best AUC={best_auc:.4f}")

    # =============================================================================
    # CSV Logging
    # =============================================================================

    if not os.path.exists(LOG_PATH):

        with open(LOG_PATH, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_accuracy",
                "val_auc",
                "is_best",
                "class_counts",
                "timestamp"
            ])

    # =============================================================================
    # Training Loop
    # =============================================================================

    for epoch in range(start_epoch, NUM_EPOCHS):

        print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}")

        # ---------------------------------------------------
        # Trigger per-epoch subsampling
        # ---------------------------------------------------

        train_dataset.on_epoch_start()

        if SUBSAMPLE_PER_EPOCH is not None:
            print(f"[INFO] Subsampled train samples: {len(train_dataset)}")
        else:
            print(f"[INFO] subsampling disabled, using: {len(train_dataset)} samples")

        # ---------------------------------------------------
        # Collect labels
        # ---------------------------------------------------

        if MODE == "video_binary":
            labels = [s["binary_label"] for s in train_dataset.samples]
        else:
            labels = [s["primary_label"] for s in train_dataset.samples]

        counts = Counter(labels)

        print(f"[INFO] Class distribution this epoch: {counts}")

        if len(counts) < 2:
            print("[WARNING] Less than 2 classes present in subsampled epoch!")

        # ---------------------------------------------------
        # Class Weights
        # ---------------------------------------------------

        total = sum(counts.values())

        class_weights = []

        for c in sorted(counts.keys()):
            class_weights.append(total / (counts[c] + 1e-6))

        print(f"[INFO] Class counts: {counts} | weights: {class_weights}")

        weights_cpu = torch.tensor(class_weights, dtype=torch.float)

        sample_weights = [weights_cpu[label].item() for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        weights_tensor = weights_cpu.to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        # ---------------------------------------------------
        # DataLoaders
        # ---------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2  
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2
        )

        if device.type == "cuda":
            train_loader = CUDAPrefetcher(train_loader, device)
            val_loader = CUDAPrefetcher(val_loader, device)

        # ---------------------------------------------------
        # Training
        # ---------------------------------------------------

        model.train()

        running_loss = 0.0
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, (frames, labels_batch) in enumerate(pbar):

            frames = frames.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):

                outputs = model(frames)
                loss = criterion(outputs, labels_batch)

            loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()      

            if (step + 1) % ACCUMULATION_STEPS == 0:

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUMULATION_STEPS

            pbar.set_postfix(
                loss=f"{running_loss / (step+1):.4f}"
            )

        avg_loss = running_loss / max(len(train_loader), 1)

        epoch_time = time.time() - epoch_start

        print(f"[INFO] Epoch time: {epoch_time/60:.2f} minutes")

        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        model.eval()

        val_metrics = evaluate(
            model,
            val_loader,
            device
        )

        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]

        print(
            f"[VAL] TrainLoss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f}"
        )

        # ---------------------------------------------------
        # Save Best Model
        # ---------------------------------------------------

        is_best = False

        if val_auc > best_auc:

            best_auc = val_auc
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc
            }, BEST_MODEL_PATH)

            print(f"[SAVE] ✨ Best model saved (AUC={best_auc:.4f})")

        # ---------------------------------------------------
        # Save Checkpoint
        # ---------------------------------------------------

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc
        }, CHECKPOINT_PATH)

        print("🔶 Checkpoint saved")

        # ---------------------------------------------------
        # CSV Logging
        # ---------------------------------------------------

        with open(LOG_PATH, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                round(avg_loss, 6),
                round(val_acc, 6),
                round(val_auc, 6),
                int(is_best),
                dict(counts),
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

    print("\n✅ Training completed successfully.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()