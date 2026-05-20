# =============================================================================
# fuse_learned.py  (FIXED — train on val, evaluate on test)
#
# KEY FIX vs original:
#   - Original ran 5-fold CV on a single pool mixing all splits.
#     This leaks test data into training of the fusion head.
#   - This version trains LR and MLP on the VAL aligned split,
#     evaluates on the TEST aligned split.
#   - 5-fold CV is kept only on VAL for hyperparameter selection.
#   - Final reported numbers come exclusively from the TEST split.
#
# Run align_scores.py twice first:
#   python align_scores.py  (SPLIT="val"  → aligned/val/)
#   python align_scores.py  (SPLIT="test" → aligned/test/)
#
# Outputs (in OUTPUT_DIR):
#   learned_fusion_results.json
#   lr_fusion_test_scores.npy
#   mlp_fusion_test_scores.npy
#   lr_model.pkl
#   mlp_model.pkl
#   roc_comparison.png
# =============================================================================

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, classification_report
)
from collections import defaultdict

# =============================================================================
# PATHS
# =============================================================================

VAL_ALIGNED_DIR  = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\aligned_alpha_val_outputs"
)
TEST_ALIGNED_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\aligned_alpha_test_outputs"
)
OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\results\learned"
)

N_FOLDS      = 5
RANDOM_STATE = 42

# =============================================================================


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def find_optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return thresholds[np.argmax(tpr - fpr)]


def print_full_metrics(tag, labels, scores):
    auc     = roc_auc_score(labels, scores)
    eer     = compute_eer(labels, scores)
    opt_thr = find_optimal_threshold(labels, scores)
    preds   = (scores >= opt_thr).astype(int)
    acc     = accuracy_score(labels, preds)
    cm      = confusion_matrix(labels, preds)
    report  = classification_report(labels, preds, digits=4)

    print(f"\n{'='*58}")
    print(f" {tag}")
    print(f"{'='*58}")
    print(f"  AUC                    : {auc:.4f}")
    print(f"  EER                    : {eer:.4f}")
    print(f"  Optimal Threshold      : {opt_thr:.4f}")
    print(f"  Accuracy @ Optimal Thr : {acc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")
    return float(auc), float(eer)


def select_lr_C(X_val, y_val, n_folds, random_state):
    """Pick best C for LR via CV on val split only."""
    print(f"\n  [LR] CV hyperparameter search on val split ({n_folds} folds) ...")
    param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    base = LogisticRegression(max_iter=1000, solver="lbfgs",
                              random_state=random_state)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_val)
    gs = GridSearchCV(base, param_grid,
                      cv=StratifiedKFold(n_folds, shuffle=True,
                                         random_state=random_state),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(X_s, y_val)
    best_C = gs.best_params_["C"]
    print(f"  [LR] Best C = {best_C}  (CV AUC = {gs.best_score_:.4f})")
    return best_C, scaler


def per_attack_aucs(labels, scores, meta):
    """Return {attack: auc} pooling real samples against each attack."""
    real_l, real_s = [], []
    atk_bkt = defaultdict(lambda: {"l": [], "s": []})
    for i, m in enumerate(meta):
        if labels[i] == 0:
            real_l.append(labels[i])
            real_s.append(scores[i])
        else:
            atk_bkt[m["attack"]]["l"].append(labels[i])
            atk_bkt[m["attack"]]["s"].append(scores[i])

    real_l = np.array(real_l)
    real_s = np.array(real_s)
    result = {}
    for atk, bkt in atk_bkt.items():
        l = np.concatenate([real_l, bkt["l"]])
        s = np.concatenate([real_s, bkt["s"]])
        result[atk] = float(roc_auc_score(l, s))
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load val aligned split  (used for fusion head training)
    # -------------------------------------------------------------------------
    print("[INFO] Loading VAL aligned split ...")
    val_vid = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_video_scores.npy"))
    val_aud = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_audio_scores.npy"))
    val_lbl = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_labels.npy"))
    with open(os.path.join(VAL_ALIGNED_DIR, "aligned_meta.json")) as f:
        val_meta = json.load(f)
    print(f"  Val  — real: {(val_lbl==0).sum()}  fake: {(val_lbl==1).sum()}  "
          f"total: {len(val_lbl)}")

    X_val = np.stack([val_aud, val_vid], axis=1).astype(np.float32)
    y_val = val_lbl.astype(int)

    # -------------------------------------------------------------------------
    # Load test aligned split  (used only for final evaluation)
    # -------------------------------------------------------------------------
    print("\n[INFO] Loading TEST aligned split ...")
    tst_vid = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_video_scores.npy"))
    tst_aud = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_audio_scores.npy"))
    tst_lbl = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_labels.npy"))
    with open(os.path.join(TEST_ALIGNED_DIR, "aligned_meta.json")) as f:
        tst_meta = json.load(f)
    print(f"  Test — real: {(tst_lbl==0).sum()}  fake: {(tst_lbl==1).sum()}  "
          f"total: {len(tst_lbl)}")

    X_tst = np.stack([tst_aud, tst_vid], axis=1).astype(np.float32)
    y_tst = tst_lbl.astype(int)

    # -------------------------------------------------------------------------
    # Baselines on TEST split
    # -------------------------------------------------------------------------
    print("\n========== Baselines (TEST split) ==========")
    aud_auc, aud_eer = print_full_metrics("Audio Only — TEST", tst_lbl, tst_aud)
    vid_auc, vid_eer = print_full_metrics("Video Only — TEST", tst_lbl, tst_vid)

    # -------------------------------------------------------------------------
    # Method 1: Logistic Regression
    #   - Select C via CV on val
    #   - Train on full val
    #   - Evaluate on test
    # -------------------------------------------------------------------------
    print("\n========== Logistic Regression ==========")
    best_C, lr_scaler = select_lr_C(X_val, y_val, N_FOLDS, RANDOM_STATE)

    X_val_s = lr_scaler.transform(X_val)
    X_tst_s = lr_scaler.transform(X_tst)

    lr_model = LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs",
                                  random_state=RANDOM_STATE)
    lr_model.fit(X_val_s, y_val)

    lr_coef_audio = float(lr_model.coef_[0][0])
    lr_coef_video = float(lr_model.coef_[0][1])
    lr_bias       = float(lr_model.intercept_[0])
    print(f"  LR coef — audio: {lr_coef_audio:.4f}  "
          f"video: {lr_coef_video:.4f}  bias: {lr_bias:.4f}")
    # Implied alpha (audio weight in linear combination):
    total = abs(lr_coef_audio) + abs(lr_coef_video)
    print(f"  Implied audio weight : {abs(lr_coef_audio)/total:.3f}  "
          f"video weight : {abs(lr_coef_video)/total:.3f}")

    lr_tst_scores = lr_model.predict_proba(X_tst_s)[:, 1].astype(np.float32)
    lr_auc, lr_eer = print_full_metrics(
        f"Logistic Regression (C={best_C}) — TEST", tst_lbl, lr_tst_scores)

    # -------------------------------------------------------------------------
    # Method 2: MLP  (simple grid over hidden sizes, CV on val)
    # -------------------------------------------------------------------------
    print("\n========== MLP ==========")
    mlp_scaler = StandardScaler()
    X_val_ms = mlp_scaler.fit_transform(X_val)
    X_tst_ms = mlp_scaler.transform(X_tst)

    hidden_candidates = [(16, 8), (32, 16), (64, 32)]
    print(f"  CV over hidden sizes {hidden_candidates} on val ...")
    best_hidden, best_cv_auc = None, -1.0

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for hs in hidden_candidates:
        fold_aucs = []
        for tr, vl in skf.split(X_val_ms, y_val):
            clf = MLPClassifier(
                hidden_layer_sizes=hs, activation="relu", solver="adam",
                max_iter=500, random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
            )
            clf.fit(X_val_ms[tr], y_val[tr])
            fold_aucs.append(roc_auc_score(y_val[vl], clf.predict_proba(X_val_ms[vl])[:, 1]))
        mean_auc = float(np.mean(fold_aucs))
        print(f"    hidden={hs}  CV AUC={mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
        if mean_auc > best_cv_auc:
            best_cv_auc  = mean_auc
            best_hidden  = hs

    print(f"  Best hidden size: {best_hidden}")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=best_hidden, activation="relu", solver="adam",
        max_iter=500, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )
    mlp_model.fit(X_val_ms, y_val)

    mlp_tst_scores = mlp_model.predict_proba(X_tst_ms)[:, 1].astype(np.float32)
    mlp_auc, mlp_eer = print_full_metrics(
        f"MLP {best_hidden} — TEST", tst_lbl, mlp_tst_scores)

    # -------------------------------------------------------------------------
    # Per-attack breakdown on TEST (best learned method)
    # -------------------------------------------------------------------------
    best_scores = lr_tst_scores if lr_auc >= mlp_auc else mlp_tst_scores
    best_tag    = "LR" if lr_auc >= mlp_auc else "MLP"
    print(f"\n========== Per-Attack ({best_tag} Fusion — TEST split) ==========")
    atk_results = per_attack_aucs(tst_lbl, best_scores, tst_meta)
    for atk in sorted(atk_results):
        print(f"  [{atk:<25}]  AUC: {atk_results[atk]:.4f}")

    # -------------------------------------------------------------------------
    # Save models
    # -------------------------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, "lr_model.pkl"),  "wb") as f:
        pickle.dump({"model": lr_model,  "scaler": lr_scaler},  f)
    with open(os.path.join(OUTPUT_DIR, "mlp_model.pkl"), "wb") as f:
        pickle.dump({"model": mlp_model, "scaler": mlp_scaler}, f)
    print("\n[INFO] Saved lr_model.pkl and mlp_model.pkl")

    np.save(os.path.join(OUTPUT_DIR, "lr_fusion_test_scores.npy"),  lr_tst_scores)
    np.save(os.path.join(OUTPUT_DIR, "mlp_fusion_test_scores.npy"), mlp_tst_scores)

    # -------------------------------------------------------------------------
    # ROC comparison plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    configs = [
        (tst_aud,        f"Audio Only  (AUC={aud_auc:.4f})", "orange",    "--"),
        (tst_vid,        f"Video Only  (AUC={vid_auc:.4f})", "green",     "--"),
        (lr_tst_scores,  f"LR Fusion   (AUC={lr_auc:.4f})",  "royalblue", "-"),
        (mlp_tst_scores, f"MLP Fusion  (AUC={mlp_auc:.4f})", "crimson",   "-"),
    ]
    for scores, label, color, ls in configs:
        fpr, tpr, _ = roc_curve(tst_lbl, scores)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — TEST split (no leakage)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_comparison.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved ROC plot → {roc_path}")

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "note": "All AUC/EER values are on the TEST split. "
                "Fusion heads trained on VAL split only.",
        "audio_only":  {"auc": aud_auc, "eer": aud_eer},
        "video_only":  {"auc": vid_auc, "eer": vid_eer},
        "lr_fusion":   {"auc": lr_auc,  "eer": lr_eer,
                        "best_C": best_C,
                        "coef_audio": lr_coef_audio,
                        "coef_video": lr_coef_video},
        "mlp_fusion":  {"auc": mlp_auc, "eer": mlp_eer,
                        "best_hidden": list(best_hidden)},
        "per_attack_best_learned": atk_results,
    }
    with open(os.path.join(OUTPUT_DIR, "learned_fusion_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Final summary table
    # -------------------------------------------------------------------------
    print("\n" + "="*58)
    print(" FINAL SUMMARY (TEST split — no leakage)")
    print("="*58)
    print(f"  {'Method':<30} {'AUC':>8}  {'EER':>8}")
    print(f"  {'-'*50}")
    for name, auc, eer in [
        ("Audio Only",     aud_auc, aud_eer),
        ("Video Only",     vid_auc, vid_eer),
        (f"LR Fusion (C={best_C})", lr_auc, lr_eer),
        (f"MLP Fusion {best_hidden}", mlp_auc, mlp_eer),
    ]:
        print(f"  {name:<30} {auc:>8.4f}  {eer:>8.4f}")
    print("="*58)
    print(f"\n[INFO] All results saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
