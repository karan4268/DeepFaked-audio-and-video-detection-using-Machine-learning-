# train_audio.py
# =============================================================================
# IMPORTANT: DATA LEAKAGE ISSUE IN ASVSPOOF TRAINING (FIXED)
# =============================================================================
#
# PROBLEM
# -------
# Initial training showed AUC = 1.000 at epoch 1, which is unrealistic and
# indicates a flawed experimental setup rather than a strong model.
#
# Root cause:
#   The ASVspoof dataset contains:
#       - bonafide (real speech)
#       - spoof attacks (A01–A19)
#
#   Earlier implementation used the default dataset split (train/val) without
#   controlling attack distribution. As a result:
#
#       Train set → attacks A01–A06
#       Val set   → attacks A01–A06
#
#   This allows the model to learn attack-specific artifacts and see the SAME
#   attack types during validation.
#
#   The model therefore "memorizes" attack patterns instead of learning
#   generalized spoof detection → artificially perfect AUC.
#
#
# SECOND ISSUE
# ------------
# When introducing attack filtering, only Axx attacks were selected:
#
#       allowed_attacks = ["A01", "A02", ...]
#
#   This unintentionally removed ALL bonafide samples because:
#
#       bonafide samples have attack label = "bonafide"
#
#   Result:
#       - Train set contained only spoof samples
#       - Val set became empty
#       - pos_weight → 0
#       - sampler → divide-by-zero
#
#
# FIX
# ---
# 1. Always include bonafide in both splits:
#
#       TRAIN_ATTACKS = ["bonafide", "A01", "A02", "A03", "A04"]
#       VAL_ATTACKS   = ["bonafide", "A05", "A06"]
#
# 2. Enforce attack-based split in DataLoader using allowed_attacks
#
# 3. This ensures:
#
#       - Train and Val have BOTH classes (real + spoof)
#       - Spoof attacks in Val are UNSEEN during training
#       - Model is forced to generalize instead of memorizing
#
#
# RESULT
# ------
# Before fix:
#       AUC = 1.000 (invalid, due to leakage)
#
# After fix:
#       AUC reflects true generalization (~0.85–0.95 expected)
#       EER becomes meaningful
#
#
# KEY TAKEAWAY
# ------------
# In spoof detection, splitting by dataset partitions alone is insufficient.
# You MUST control:
#       - attack types
#       - speaker identity (optional advanced)
#
# Otherwise, evaluation metrics are misleading.
# =============================================================================

import os
import time
import json
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from data_loader import ASVSpoofDataset, collate_fn
from Preprocessing.cache_audio import make_audio_sampler     

TRAIN_ATTACKS = ["bonafide", "A01", "A02", "A03", "A04"]
VAL_ATTACKS   = ["bonafide", "A05", "A06"]
# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x).view(-1)              # [FIX 3] was .squeeze(1)

            logits = logits.detach().cpu().numpy()
            y      = y.detach().cpu().numpy()

            mask   = ~np.isnan(logits)
            logits = logits[mask]
            y      = y[mask]

            if len(logits) == 0:
                continue

            all_logits.extend(logits)
            all_labels.extend(y)

    if len(all_logits) == 0:
        print("[WARNING] No valid logits during evaluation")
        return 0.5, 0.5

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    auc = roc_auc_score(all_labels, all_logits)
    eer = compute_eer(all_labels, all_logits)

    return auc, eer

# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    CACHE_BASE = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
    MODEL_DIR  = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio"

    os.makedirs(MODEL_DIR, exist_ok=True)

    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "audio_model.pth")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "audio_model_best.pth")
    LOG_PATH        = os.path.join(MODEL_DIR, "training_log.csv")

    # -------------------------------------------------------------------------
    # DATASETS  [FIX 7]
    # -------------------------------------------------------------------------
    BATCH_SIZE  = 16
    NUM_WORKERS = 2

    train_dataset = ASVSpoofDataset(
        CACHE_BASE,
        split="train",
        augment=True,
        target_length=None,
        allowed_attacks=TRAIN_ATTACKS     
    )

    val_dataset = ASVSpoofDataset(
        CACHE_BASE,
        split="val",
        augment=False,
        target_length=None,
        allowed_attacks=VAL_ATTACKS      
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples  : {len(val_dataset)}")

    # 🔍 DEBUG: attack distribution check
    train_attacks_present = set([s["attack"] for s in train_dataset.samples])
    val_attacks_present   = set([s["attack"] for s in val_dataset.samples])

    print(f"[DEBUG] Train attacks: {sorted(train_attacks_present)}")
    print(f"[DEBUG] Val attacks  : {sorted(val_attacks_present)}")

    # -------------------------------------------------------------------------
    # CLASS WEIGHT  [UNCHANGED LOGIC]
    # -------------------------------------------------------------------------
    labels    = np.array(train_dataset.labels)
    num_pos   = int(np.sum(labels == 1))
    num_neg   = int(np.sum(labels == 0))

    pos_weight = torch.tensor(
        [num_neg / max(num_pos, 1)],
        dtype=torch.float32
    ).to(device)

    print(f"[INFO] Train — real: {num_neg}  spoof: {num_pos}")
    print(f"[INFO] pos_weight  : {pos_weight.item():.4f}")

    # -------------------------------------------------------------------------
    # DATALOADERS
    # -------------------------------------------------------------------------

    # [FIX 8] use dataset.labels directly (cleaner + correct)
    sampler = make_audio_sampler(
        [{"label": int(l)} for l in train_dataset.labels]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model     = AudioResNet18(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # [FIX 5] patience raised 2 → 3 to avoid premature LR decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # [FIX 6] AMP enabled — safe with fixed-length padded mel inputs
    USE_AMP = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # -------------------------------------------------------------------------
    # RESUME CHECKPOINT
    # -------------------------------------------------------------------------
    start_epoch = 0
    best_eer    = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        print("[INFO] Loading checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_eer    = ckpt.get("best_eer", float("inf"))

        print(f"[INFO] Resumed from epoch {start_epoch} | Best EER={best_eer:.4f}")

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 20

    for epoch in range(start_epoch, NUM_EPOCHS):

        print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()

        running_loss  = 0.0
        valid_steps   = 0                           # [FIX 4]
        skipped_steps = 0                           # [FIX 4]

        for x, y in train_loader:

            if torch.isnan(x).any():
                skipped_steps += 1
                print(f"[WARNING] NaN input batch skipped ({skipped_steps} total)")
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x).view(-1)          # [FIX 3] was .squeeze(1)
                logits = torch.clamp(logits, -20, 20)
                # --- temporary NaN diagnostic, remove after fix ---
                #with torch.cuda.amp.autocast(enabled=USE_AMP):
                    #logits = model(x).view(-1)
                    #logits = torch.clamp(logits, -20, 20)

                    #if torch.isnan(logits).any():
                        #print(f"[NaN SOURCE] logits are NaN — model output bad")
                        #continue

                    #if torch.isnan(y).any():
                        #print(f"[NaN SOURCE] labels are NaN")
                        #continue

                    #print(f"[DEBUG] logits min={logits.min():.3f} max={logits.max():.3f} "
                        #f"y unique={y.unique()} pos_weight={pos_weight.item():.4f}")

                    #loss = criterion(logits, y)

                    #if torch.isnan(loss):
                        #print(f"[NaN SOURCE] loss is NaN — logits were fine, issue is in criterion")
                        #continue
                loss   = criterion(logits, y)

            if torch.isnan(loss):
                skipped_steps += 1
                print(f"[WARNING] NaN loss skipped ({skipped_steps} total)")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            valid_steps  += 1                       # [FIX 4]

        # [FIX 4] divide by actual valid steps, not len(train_loader)
        avg_loss = running_loss / max(valid_steps, 1)

        if skipped_steps > 0:
            print(f"[WARNING] {skipped_steps} batches skipped this epoch")

        # ----------------  VALIDATION  ----------------
        val_auc, val_eer = evaluate(model, val_loader, device)

        print(
            f"[VAL] Loss: {avg_loss:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"EER: {val_eer:.4f}"
        )

        scheduler.step(val_eer)

        # ----------------  SAVE BEST  ----------------
        is_best = False

        if val_eer < best_eer:
            best_eer = val_eer
            is_best  = True

            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer":             best_eer
            }, BEST_MODEL_PATH)

            print(f"[SAVE] ✨ Best model saved (EER={best_eer:.4f})")

        # ----------------  CHECKPOINT  ----------------
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eer":             best_eer
        }, CHECKPOINT_PATH)

        # ----------------  LOG  ----------------
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(avg_loss, 6),
                round(val_auc, 6),
                round(val_eer, 6),
                is_best,
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

        print("🔶 Checkpoint saved")

    print("\n[✓] Training completed successfully.")
    print(f"[INFO] Best Validation EER: {best_eer:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()