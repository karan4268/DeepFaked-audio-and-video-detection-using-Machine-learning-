# =============================================================================
# align_and_fuse_test.py (FIXED)
#
# FIXES vs previous version:
#
#   [FIX 1] FT_VIDEO_IDS now points to finetuned_video_file_ids.npy instead
#           of finetuned_video_indices.npy. The old file contained positional
#           integers [0, 1, 2, ...] which when cast to str() produced keys
#           "0", "1", "2" — these never matched any file_id in the fusion
#           master index so vid_map.get(fid) returned None for every sample,
#           missing_v = total samples, and aligned set was empty.
#
#   [FIX 2] np.load for both file_id arrays now uses allow_pickle=True.
#           String arrays saved with np.save are object arrays. numpy >= 1.16.3
#           raises "Object arrays cannot be loaded when allow_pickle=False"
#           by default, which would crash before alignment even starts.
#
#   [FIX 3] Removed unused fusion_map dict. It was constructed but never
#           referenced — alignment uses test_data + vid_map/aud_map directly.
#
#   [NOTE]  Alpha sweep and LR/MLP cross-val both operate on the test set.
#           Alpha is therefore an oracle weight (upper bound, not generalisable).
#           LR/MLP OOF scores are fit and evaluated on the same test distribution
#           (no separate val split available). Both are labelled in print output.
#           This is standard practice when a dedicated fusion-val split does not
#           exist, but results should be interpreted accordingly.
# =============================================================================

import os
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

# =============================================================================
# PATHS
# =============================================================================

FUSION_INDEX = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\fusion_master_index.json"
)

FT_VIDEO_SCORES = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_finetuned\eval_test_after_finetuning\finetuned_video_scores.npy"
)

# FIX 1: was finetuned_video_indices.npy (positional ints → keys "0","1","2"
#         which never matched any file_id → all samples dropped silently)
FT_VIDEO_IDS = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\video_finetuned\eval_test_after_finetuning\finetuned_video_file_ids.npy"
)

AUDIO_SCORES = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_fakeavceleb_test_split\audio_test_scores.npy"
)

AUDIO_IDS = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_fakeavceleb_test_split\audio_test_file_ids.npy"
)

OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\results\fusion_clean"
)

N_FOLDS      = 5
RANDOM_STATE = 42


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i   = np.argmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def find_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def print_metrics(name, y, s):
    auc  = roc_auc_score(y, s)
    eer  = compute_eer(y, s)
    thr  = find_threshold(y, s)
    pred = (s >= thr).astype(int)
    acc  = accuracy_score(y, pred)
    cm   = confusion_matrix(y, pred)

    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    print(f"AUC : {auc:.4f}")
    print(f"EER : {eer:.4f}")
    print(f"ACC : {acc:.4f}")
    print(f"THR : {thr:.4f}")
    print(f"\nCM:\n{cm}")

    return auc, eer


# =============================================================================
# CROSS VALIDATION
# NOTE: X and y here are drawn from the test set (no separate fusion-val split).
# OOF scores are fit and evaluated on the same test distribution.
# Results reflect in-distribution fusion capacity, not generalisation.
# =============================================================================

def cross_val(X, y, model_cls, kwargs):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))

    for f, (tr, va) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr    = scaler.fit_transform(X[tr])
        Xva    = scaler.transform(X[va])

        model = model_cls(**kwargs)
        model.fit(Xtr, y[tr])
        oof[va] = model.predict_proba(Xva)[:, 1]

        auc = roc_auc_score(y[va], oof[va])
        print(f"  Fold {f+1}: AUC={auc:.4f}")

    return oof


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------
    # LOAD MASTER INDEX
    # --------------------------------------------------
    print("[INFO] Loading fusion master index...")
    with open(FUSION_INDEX) as f:
        fusion_index = json.load(f)

    # FIX 3: removed unused fusion_map dict

    # --------------------------------------------------
    # LOAD VIDEO SCORES + FILE IDS
    # FIX 1: loading file_ids (strings) not positional indices
    # FIX 2: allow_pickle=True required for string object arrays
    # --------------------------------------------------
    print("[INFO] Loading video scores...")
    vid_scores = np.load(FT_VIDEO_SCORES)
    vid_ids    = np.load(FT_VIDEO_IDS, allow_pickle=True)   # FIX 2

    vid_map = {
        str(fid): float(score)
        for fid, score in zip(vid_ids, vid_scores)
    }

    # --------------------------------------------------
    # LOAD AUDIO SCORES + FILE IDS
    # FIX 2: allow_pickle=True required for string object arrays
    # --------------------------------------------------
    print("[INFO] Loading audio scores...")
    aud_scores = np.load(AUDIO_SCORES)
    aud_ids    = np.load(AUDIO_IDS, allow_pickle=True)      # FIX 2

    aud_map = {
        str(fid): float(score)
        for fid, score in zip(aud_ids, aud_scores)
    }

    print(f"  Video score entries : {len(vid_map)}")
    print(f"  Audio score entries : {len(aud_map)}")

    # --------------------------------------------------
    # ALIGN USING MASTER INDEX
    # --------------------------------------------------
    print("[INFO] Aligning using master index (is_test=True only)...")

    test_data = [x for x in fusion_index if x["is_test"]]

    X_vid, X_aud, y = [], [], []
    missing_v = 0
    missing_a = 0
    invalid_modality = 0

    for item in test_data:
        fid = item["file_id"]

        # skip entries where one modality failed to cache
        if not item["audio_valid"] or not item["video_valid"]:
            invalid_modality += 1
            continue

        v = vid_map.get(fid)
        a = aud_map.get(fid)

        if v is None:
            missing_v += 1
            continue
        if a is None:
            missing_a += 1
            continue

        X_vid.append(v)
        X_aud.append(a)
        y.append(item["label"])

    X_vid = np.array(X_vid)
    X_aud = np.array(X_aud)
    y     = np.array(y)

    print(f"\n[INFO] Test entries in master index   : {len(test_data)}")
    print(f"[INFO] Skipped (incomplete modality)  : {invalid_modality}")
    print(f"[INFO] Missing video score            : {missing_v}")
    print(f"[INFO] Missing audio score            : {missing_a}")
    print(f"[INFO] Final aligned samples          : {len(y)}")
    print(f"[INFO] Real: {(y==0).sum()} | Fake: {(y==1).sum()}")

    if len(y) == 0:
        raise RuntimeError(
            "Alignment produced 0 samples. "
            "Verify that finetuned_video_file_ids.npy and audio_test_file_ids.npy "
            "use the same file_id convention as fusion_master_index.json."
        )

    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        raise RuntimeError(
            f"Only one class present after alignment "
            f"(real={(y==0).sum()}, fake={(y==1).sum()}). Cannot compute AUC."
        )

    # --------------------------------------------------
    # BASELINES
    # --------------------------------------------------
    print("\n========== BASELINES ==========")
    print_metrics("Audio Only", y, X_aud)
    print_metrics("Video Only", y, X_vid)

    # --------------------------------------------------
    # WEIGHTED FUSION
    # NOTE: alpha is selected on the test set itself — oracle upper bound.
    # --------------------------------------------------
    print("\n[INFO] Sweeping alpha (audio weight)...")
    print("[NOTE] Alpha selected on test set — treat as oracle, not generalised result.")

    best_auc = 0
    best_a   = 0.0

    for a in np.arange(0.0, 1.05, 0.05):
        fused = a * X_aud + (1 - a) * X_vid
        auc   = roc_auc_score(y, fused)

        if auc > best_auc:
            best_auc = auc
            best_a   = a

        print(f"  alpha={a:.2f}  AUC={auc:.4f}")

    print(f"\nBest alpha: {best_a:.2f}  AUC={best_auc:.4f}  [oracle]")

    best_fused = best_a * X_aud + (1 - best_a) * X_vid
    print_metrics(f"Weighted Fusion (alpha={best_a:.2f}, oracle)", y, best_fused)

    # save best weighted fusion scores
    np.save(os.path.join(OUTPUT_DIR, "weighted_fusion_scores.npy"), best_fused)
    np.save(os.path.join(OUTPUT_DIR, "fusion_labels.npy"),          y)

    # --------------------------------------------------
    # LEARNED FUSION
    # NOTE: CV fit and evaluated on test set — no separate fusion-val split.
    # OOF AUC reflects in-distribution fusion capacity only.
    # --------------------------------------------------
    X = np.stack([X_aud, X_vid], axis=1)

    print("\n========== Logistic Regression (CV on test set) ==========")
    print("[NOTE] No separate fusion-val split — OOF results are in-distribution only.")
    lr_oof = cross_val(X, y, LogisticRegression, dict(max_iter=1000))
    print_metrics("LR Fusion (OOF)", y, lr_oof)
    np.save(os.path.join(OUTPUT_DIR, "lr_fusion_oof_scores.npy"), lr_oof)

    print("\n========== MLP (CV on test set) ==========")
    print("[NOTE] No separate fusion-val split — OOF results are in-distribution only.")
    mlp_oof = cross_val(X, y, MLPClassifier,
                        dict(hidden_layer_sizes=(16, 8), max_iter=500, random_state=RANDOM_STATE))
    print_metrics("MLP Fusion (OOF)", y, mlp_oof)
    np.save(os.path.join(OUTPUT_DIR, "mlp_fusion_oof_scores.npy"), mlp_oof)

    print(f"\n[INFO] All outputs saved → {OUTPUT_DIR}")
    print("[INFO] DONE.")


if __name__ == "__main__":
    main()