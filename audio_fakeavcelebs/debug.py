# ============================================================
# debug_audio_pipeline.py
# Central Debug Suite for Audio Deepfake Pipeline
# ============================================================

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# METRICS
# ============================================

def compute_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


# ============================================
# 1. DATA INSPECTION
# ============================================

def inspect_loader(loader):
    print("\n[INSPECTING DATA]\n")

    for i, (x, y) in enumerate(loader):

        print(f"Batch {i}")
        print("x shape:", x.shape)
        print("y:", y[:10])
        print("x mean:", x.mean().item())
        print("x std:", x.std().item())

        if i == 2:
            break


# ============================================
# 2. BATCH DEBUG (NEW)
# ============================================

def debug_batch(model, loader):
    print("\n[DEBUG BATCH]\n")

    model.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)

            print("logits shape:", logits.shape)
            print("probs shape :", probs.shape)
            print("labels shape:", y.shape)

            print("logits sample:", logits[:5].cpu().numpy())
            print("probs sample :", probs[:5].cpu().numpy())
            print("labels sample:", y[:5].cpu().numpy())

            break  # only first batch


# ============================================
# 3. FINAL OUTPUT DEBUG (NEW)
# ============================================

def debug_final(model, loader):
    print("\n[DEBUG FINAL]\n")

    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    print("Final scores shape:", all_scores.shape)
    print("Final labels shape:", all_labels.shape)
    print("Scores range:", (all_scores.min(), all_scores.max()))

    auc = compute_auc(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    print(f"[FINAL METRICS] AUC: {auc:.4f} | EER: {eer:.4f}")


# ============================================
# 4. LEAKAGE TESTS (YOUR ORIGINAL)
# ============================================

def run_test(model, loader, mode="normal"):

    model.eval()

    y_true = []
    y_score = []

    with torch.no_grad():
        for x, y in tqdm(loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # ------------------------------
            # TEST MODES
            # ------------------------------
            if mode == "shuffle_labels":
                y = y[torch.randperm(len(y))]

            elif mode == "zero_input":
                x = torch.zeros_like(x)

            elif mode == "random_input":
                x = torch.randn_like(x)

            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)

            y_true.extend(y.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    return compute_auc(y_true, y_score)


def debug_pipeline(model, loader):
    print("\n================ DEBUG START ================\n")

    auc_normal = run_test(model, loader, "normal")
    print(f"[NORMAL] AUC: {auc_normal:.4f}")

    auc_shuffle = run_test(model, loader, "shuffle_labels")
    print(f"[SHUFFLE LABELS] AUC: {auc_shuffle:.4f}")

    auc_zero = run_test(model, loader, "zero_input")
    print(f"[ZERO INPUT] AUC: {auc_zero:.4f}")

    auc_random = run_test(model, loader, "random_input")
    print(f"[RANDOM INPUT] AUC: {auc_random:.4f}")

    print("\n================ ANALYSIS ================\n")

    if auc_shuffle > 0.7:
        print("🚨 Leakage: Model ignores labels")

    if auc_zero > 0.7:
        print("🚨 Leakage: Labels embedded in input pipeline")

    if auc_random > 0.7:
        print("🚨 Severe leakage or dataset bias")

    if auc_normal > 0.95:
        print("⚠️ Suspiciously high performance")

    print("\n================ END =====================\n")