# =============================================================================
# train_video_model.py
# Training script for Deepfake Video Detection using VideoModel (ResNet18)
#
# IMPORTANT:
# - Run cache_face.py ONCE before training
# - Training loads faces from cached_faces/
#
# IMPROVEMENTS ADDED:
# Class imbalance handling (Weighted Loss + Sampler)
# Mixed Precision Training (AMP)
# Learning Rate Scheduler
# Label Smoothing
# Better GPU utilization
#
# Date: 25-03-26
# =============================================================================

import os
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from Preprocessing.video_preprocessing import DeepfakedDataset
from video_model import VideoModel
from eval_video import evaluate


def main():
    # =============================================================================
    # Device
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =============================================================================
    # Paths
    # =============================================================================
    ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video"

    CHECKPOINT_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\video_model.pth_3D_CNN-ResNet18"
    BEST_MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\video_model_best_3D_CNN-ResNet18.pth"
    LOG_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\training_log_3D_ResNet18_FF++c23.csv"

    # =============================================================================
    # Dataset
    # =============================================================================
    FRAMES_PER_VIDEO = 24 # 24 FPS from original video (adjust based on GPU VRAM and model capacity)
    BATCH_SIZE = 3   # (use AMP to fit) 4 was at 96% GPU VRAM, 3 is safer for 1650Ti with some margin for OS and other processes. Adjust based on GPU Specs.
    NUM_WORKERS = 2 # CPU cores for data loading (adjust based on your CPU)

    train_dataset = DeepfakedDataset(
        root_dir=ROOT_DIR,
        split="train",
        frames_per_video=FRAMES_PER_VIDEO,
        device=device.type
    )

    val_dataset = DeepfakedDataset(
        root_dir=ROOT_DIR,
        split="val",
        frames_per_video=FRAMES_PER_VIDEO,
        device=device.type
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples  : {len(val_dataset)}")

    # =============================================================================
    # Class Imbalance Handling
    # =============================================================================
    print("[INFO] Computing class weights...")

    labels = train_dataset.labels  # dataset must expose labels list
    real_count = labels.count(0)
    fake_count = labels.count(1)

    print(f"[INFO] Real: {real_count} | Fake: {fake_count}")

    class_weights = torch.tensor([
        fake_count / real_count,  # weight for class 0 (real)
        1.0
    ]).to(device)

    # Weighted sampler (balanced batches)
    sample_weights = [1.0 / (real_count if l == 0 else fake_count) for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # =============================================================================
    # Model
    # =============================================================================
    model = VideoModel().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05  # 🔥 improves generalization
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15
    )

    scaler = torch.cuda.amp.GradScaler()

    # =============================================================================
    # Resume from checkpoint
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
                "timestamp"
            ])

    # =============================================================================
    # Training Loop
    # =============================================================================
    NUM_EPOCHS = 15

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n[INFO] Epoch {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        running_loss = 0.0

        # -------- TRAIN LOOP --------
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(): # AMP for faster training and larger batches saves GPU VRAM
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)

        # -------- VALIDATION --------
        model.eval()
        val_acc, val_auc = evaluate(
                                    model,
                                    val_loader,
                                    device,
                                    tta=1   # important → keep validation stable
                                )

        print(
            f"[VAL] TrainLoss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f}"
        )

        # -------- SAVE BEST --------
        is_best = False
        if val_auc > best_auc + 1e-4: # small threshold to avoid saving for negligible improvements
            best_auc = val_auc
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc
            }, BEST_MODEL_PATH)

            print(f"[SAVE] ✨ Best model saved (AUC={best_auc:.4f})")

        # -------- SAVE CHECKPOINT --------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc
        }, CHECKPOINT_PATH)

        print("🔶 Checkpoint saved")

        # -------- Scheduler Step --------
        scheduler.step()

        # -------- LOG TO CSV --------
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(avg_loss, 6),
                round(val_acc, 6),
                round(val_auc, 6),
                int(is_best),
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

    print("\n✅ Training completed successfully.")


# =============================================================================
# Windows multiprocessing entry point
# =============================================================================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()