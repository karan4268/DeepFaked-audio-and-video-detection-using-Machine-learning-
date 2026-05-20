# train_audio.py
# -------------------------------------------------------
# Train on ASVspoof LA + ITW (balanced sampling)
# Validation: ASVspoof (A05/A06) + ITW (val split)
#
# FIXES vs previous version:
#   [FIX 1] Save / log block was OUTSIDE the epoch for-loop (indentation
#           error). Only the last epoch's metrics were ever checked for
#           best-model saving, and the CSV only got one row. Fixed by
#           moving everything inside the loop at the correct indent level.
#
#   [FIX 2] Best model was being saved to CHECKPOINT_PATH, then immediately
#           overwritten by a raw state_dict save to the same path.
#           BEST_MODEL_PATH was never written to. Fixed: best saves to
#           BEST_MODEL_PATH (full dict), checkpoint saves to CHECKPOINT_PATH
#           (full dict) once per epoch, no double-save.
#
#   [FIX 3] No resume logic. Added checkpoint loading at startup, matching
#           the pattern from the original single-dataset train_audio.py.
#
#   [FIX 4] BCEWithLogitsLoss had no pos_weight. With the weighted sampler
#           the class balance is roughly even, but pos_weight gives an extra
#           gradient push toward real/fake calibration. Computed from the
#           filtered training samples.
#
#   [FIX 5] scheduler.step() was inside the loop but BEFORE the save block,
#           which was outside the loop — so the scheduler only stepped once
#           total (after epoch 20). Now both are correctly inside the loop.
#
# IMPROVEMENTS:
#   [NEW 1] Label smoothing (eps=0.05) on BCEWithLogitsLoss.
#           Prevents the model from pushing logits to ±∞ on confident
#           predictions, improving calibration on unseen domains (FakeAVCeleb).
#
#   [NEW 2] CosineAnnealingWarmRestarts (T_0=7) instead of plain cosine.
#           Periodic LR resets help escape sharp minima that don't transfer
#           cross-domain.
#
#   [NEW 3] Save criterion changed from itw_eer-only to combined_eer.
#           The fusion model will run on FakeAVCeleb which is closer to ITW
#           in distribution, but we still need the model to not collapse on
#           ASVspoof. combined_eer = (asv_eer + itw_eer) / 2 balances both.
#
#   [NEW 4] Early stopping added (patience=5 on combined_eer) to prevent
#           overfitting on the last few epochs.
# -------------------------------------------------------

import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from data_loader import AudioDataset, collate_fn


# -------------------------------------------------------
# ATTACK CONFIG
# -------------------------------------------------------
ASV_TRAIN_ATTACKS = ["bonafide", "A01", "A02", "A03", "A04"]
ASV_VAL_ATTACKS   = ["bonafide", "A05", "A06"]

# -------------------------------------------------------
# LABEL SMOOTHING  [NEW 1]
# Soft targets: real → 0.025, fake → 0.975 (with eps=0.05)
# Prevents logit saturation and improves cross-domain calibration.
# -------------------------------------------------------
def smooth_bce(logits: torch.Tensor, targets: torch.Tensor,
               pos_weight: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    targets_smooth = targets * (1.0 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(
        logits, targets_smooth, pos_weight=pos_weight
    )


# -------------------------------------------------------
# METRICS
# -------------------------------------------------------
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).view(-1).cpu().numpy()
            labels = y.numpy()

            mask = ~np.isnan(logits)
            all_logits.extend(logits[mask])
            all_labels.extend(labels[mask])

    if len(all_logits) == 0:
        print("[WARNING] No valid logits during evaluation — returning 0.5")
        return 0.5, 0.5

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    if len(np.unique(all_labels)) < 2:
        print("[WARNING] Only one class in eval set — AUC/EER undefined")
        return 0.5, 0.5

    auc = roc_auc_score(all_labels, all_logits)
    eer = compute_eer(all_labels, all_logits)
    return auc, eer


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    CACHE_BASE = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
    MODEL_DIR  = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio"
    os.makedirs(MODEL_DIR, exist_ok=True)

    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "audio_combined.pth")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "audio_combined_best.pth")
    LOG_PATH        = os.path.join(MODEL_DIR, "training_log_combined.csv")

    BATCH_SIZE  = 16
    NUM_WORKERS = 2
    NUM_EPOCHS  = 20
    EARLY_STOP_PATIENCE = 5          
    ASV_EER_THRESHOLD   = 0.35       # safeguard: don't save if ASV collapses

    # ---------------------------------------------------
    # TRAIN DATASET
    # The AudioDataset now applies allowed_attacks only to asvspoof entries,
    # so ITW fake samples are never dropped by the ASV attack filter.
    # ---------------------------------------------------
    train_dataset = AudioDataset(
        CACHE_BASE,
        splits=["train", "itw_train"],
        augment=True,
        allowed_attacks=ASV_TRAIN_ATTACKS   # ITW entries always pass [FIX 1 in data_loader]
    )

    # ---------------------------------------------------
    # SAMPLER  (class-balanced × dataset-balanced)
    # ---------------------------------------------------
    labels   = np.array([s["label"]   for s in train_dataset.samples])
    datasets = np.array([s["dataset"] for s in train_dataset.samples])

    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    w_class       = class_weights[labels]

    unique_ds  = list(set(datasets))
    ds_counts  = {d: int((datasets == d).sum()) for d in unique_ds}
    ds_weights = {d: 1.0 / ds_counts[d] for d in unique_ds}
    w_dataset  = np.array([ds_weights[d] for d in datasets])

    sample_weights = w_class * w_dataset

    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Real: {np.sum(labels==0)} | Fake: {np.sum(labels==1)}")
    for d in unique_ds:
        print(f"[INFO]   {d}: {ds_counts[d]} samples")

    # ---------------------------------------------------
    # VALIDATION DATASETS
    # ---------------------------------------------------
    val_asv = AudioDataset(
        CACHE_BASE,
        splits=["val"],
        augment=False,
        allowed_attacks=ASV_VAL_ATTACKS
    )

    val_itw = AudioDataset(
        CACHE_BASE,
        splits=["itw_val"],
        augment=False
    )

    print(f"[INFO] Val ASVspoof: {len(val_asv)} | Val ITW: {len(val_itw)}")

    # ---------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )

    val_asv_loader = DataLoader(
        val_asv,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )

    val_itw_loader = DataLoader(
        val_itw,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    model = AudioResNet18(pretrained=True).to(device)

    # [FIX 4] pos_weight from filtered training labels
    num_pos    = int(np.sum(labels == 1))
    num_neg    = int(np.sum(labels == 0))
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
    print(f"[INFO] pos_weight: {pos_weight.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # [NEW 2] Warm restarts: re-explore loss landscape every 7 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=7, T_mult=1, eta_min=1e-6
    )

    USE_AMP = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ---------------------------------------------------
    # [FIX 3] RESUME FROM CHECKPOINT
    # ---------------------------------------------------
    start_epoch = 0
    best_eer    = float("inf")
    no_improve  = 0

    if os.path.exists(CHECKPOINT_PATH):
        print("[INFO] Loading checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer    = ckpt.get("best_eer", float("inf"))
        print(f"[INFO] Resumed from epoch {start_epoch} | Best EER={best_eer:.4f}")

    # ---------------------------------------------------
    # CSV LOG HEADER
    # ---------------------------------------------------
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss",
                "asv_auc", "asv_eer",
                "itw_auc", "itw_eer",
                "combined_eer",
                "lr",
                "is_best",
                "timestamp"
            ])

    # ---------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        model.train()
        running_loss  = 0.0
        steps         = 0
        skipped       = 0

        for x, y, _ in train_loader:
            if torch.isnan(x).any():
                skipped += 1
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x).view(-1)
                logits = torch.clamp(logits, -20, 20)
                # [NEW 1] Label smoothing via smooth_bce
                loss = smooth_bce(logits, y, pos_weight)

            if torch.isnan(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps        += 1

        if skipped > 0:
            print(f"[WARNING] {skipped} batches skipped (NaN)")

        avg_loss = running_loss / max(steps, 1)

        # [FIX 5] scheduler.step() is now inside the loop
        scheduler.step()

        # ---------------- VALIDATION ----------------
        asv_auc, asv_eer = evaluate(model, val_asv_loader, device)
        itw_auc, itw_eer = evaluate(model, val_itw_loader, device)

        # [NEW 3] Save on combined EER, not itw_eer alone
        combined_eer = (asv_eer + itw_eer) / 2

        print(
            f"[VAL] Loss: {avg_loss:.4f} | "
            f"ASV → AUC: {asv_auc:.4f}  EER: {asv_eer:.4f} | "
            f"ITW → AUC: {itw_auc:.4f}  EER: {itw_eer:.4f} | "
            f"Combined EER: {combined_eer:.4f}"
        )

        # ---------------- SAVE BEST  [FIX 1, FIX 2] ----------------
        # Both save blocks are now INSIDE the epoch loop at the correct
        # indentation level. BEST_MODEL_PATH and CHECKPOINT_PATH are
        # written to separately — no more overwriting the full checkpoint
        # with a bare state_dict.
        is_best = False

        if combined_eer < best_eer and asv_eer < ASV_EER_THRESHOLD:
            best_eer   = combined_eer
            is_best    = True
            no_improve = 0

            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer":             best_eer,
                "asv_eer":              asv_eer,
                "itw_eer":              itw_eer
            }, BEST_MODEL_PATH)   # [FIX 2] save best to BEST_MODEL_PATH

            print(f"[SAVE] ✨ Best model saved  "
                  f"combined_EER={best_eer:.4f}  "
                  f"ASV_EER={asv_eer:.4f}  ITW_EER={itw_eer:.4f}")
        else:
            no_improve += 1

        # Checkpoint every epoch (for resume)  [FIX 2]
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eer":             best_eer
        }, CHECKPOINT_PATH)
        print("🔶 Checkpoint saved")

        # ---------------- CSV LOG  [FIX 1] ----------------
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(avg_loss, 6),
                round(asv_auc, 6), round(asv_eer, 6),
                round(itw_auc, 6), round(itw_eer, 6),
                round(combined_eer, 6),
                round(optimizer.param_groups[0]["lr"], 8),
                int(is_best),
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

        # ---------------- EARLY STOPPING  [NEW 4] ----------------
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. "
                  f"Best combined EER: {best_eer:.4f}")
            break

    print(f"\n✅ Training complete. Best combined EER: {best_eer:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()