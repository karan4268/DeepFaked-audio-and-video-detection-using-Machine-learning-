# =============================================================================
# train_audio_model.py (CLEAN + CENTRALIZED DEBUG)
# =============================================================================

import os
import time
import csv
import json
import random
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

from audio_model import AudioResNet18
from data_loader import AudioDataset


import debug

# =============================================================================
# DEBUG FLAGS
# =============================================================================
DEBUG_MODE = True
DEBUG_SANITY_RUN = True   # leakage tests (slow)

# =============================================================================
# CONFIG
# =============================================================================

JSON_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2\audio_dataset_split_fixed.json"
AUDIO_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"

MODEL_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "audio_model_best.pth")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 20

SUBSAMPLE_PER_EPOCH = None  # set 4000 if needed


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

    all_labels, all_scores = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = roc_auc_score(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    return auc, eer


# =============================================================================
# LOAD SPLIT
# =============================================================================

def load_split():
    with open(JSON_PATH) as f:
        split = json.load(f)
    return split["train"], split["val"]


# =============================================================================
# MAIN
# =============================================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD SPLITS
    # ------------------------------------------------------------

    full_train_samples, val_samples = load_split()

    # 🔍 leakage check
    train_ids = set([x["path"] for x in full_train_samples])
    val_ids = set([x["path"] for x in val_samples])

    overlap = train_ids & val_ids
    print("[CHECK] Overlap:", len(overlap))

    # ------------------------------------------------------------
    # VAL LOADER
    # ------------------------------------------------------------

    val_dataset = AudioDataset(val_samples, AUDIO_ROOT, augment=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("[INFO] Val loader ready")

    # ✅ DEBUG: DATA INSPECTION
    if DEBUG_MODE:
        debug.inspect_loader(val_loader)

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = AudioResNet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ✅ DEBUG: PRE-TRAIN LEAKAGE CHECK
    if DEBUG_SANITY_RUN:
        debug.debug_pipeline(model, val_loader)

    # ------------------------------------------------------------
    # LOG FILE
    # ------------------------------------------------------------

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_auc", "val_eer", "time_sec"])

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):

        print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}")

        # --------------------------------------------------------
        # SUBSAMPLING
        # --------------------------------------------------------

        if SUBSAMPLE_PER_EPOCH:
            train_samples = random.sample(
                full_train_samples,
                min(SUBSAMPLE_PER_EPOCH, len(full_train_samples))
            )
        else:
            train_samples = full_train_samples

        # --------------------------------------------------------
        # CLASS DISTRIBUTION
        # --------------------------------------------------------

        labels = [s["binary_label"] for s in train_samples]
        counts = Counter(labels)

        print("[INFO] Class distribution:", counts)

        num_pos = counts[1]
        num_neg = counts[0]

        pos_weight = torch.tensor(
            num_neg / (num_pos + 1e-6),
            device=device
        )

        print(f"[INFO] pos_weight: {pos_weight.item():.4f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # --------------------------------------------------------
        # WEIGHTED SAMPLER
        # --------------------------------------------------------

        weights = {
            0: 1.0 / (counts[0] + 1e-6),
            1: 1.0 / (counts[1] + 1e-6)
        }

        sample_weights = [weights[l] for l in labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # --------------------------------------------------------
        # TRAIN LOADER
        # --------------------------------------------------------

        train_dataset = AudioDataset(train_samples, AUDIO_ROOT, augment=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        # --------------------------------------------------------
        # TRAINING
        # --------------------------------------------------------

        model.train()
        running_loss = 0.0
        start_time = time.time()

        for x, y in tqdm(train_loader):

            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val_auc, val_eer = evaluate(model, val_loader, device)

        # ✅ DEBUG: POST-EPOCH ANALYSIS
        if DEBUG_MODE:
            debug.debug_batch(model, val_loader)
            debug.debug_final(model, val_loader)

        epoch_time = time.time() - start_time

        print(f"[VAL] Loss: {avg_loss:.4f} | AUC: {val_auc:.4f} | EER: {val_eer:.4f}")

        # --------------------------------------------------------
        # SAVE BEST
        # --------------------------------------------------------

        if val_auc > best_auc:
            best_auc = val_auc

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc
            }, BEST_MODEL_PATH)

            print(f"[SAVE]✨ Best model (AUC={best_auc:.4f})")

        # --------------------------------------------------------
        # LOG
        # --------------------------------------------------------

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, val_auc, val_eer, epoch_time])


# =============================================================================

if __name__ == "__main__":
    main()