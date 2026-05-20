# =============================================================================
# fuse_weighted.py  (FIXED — tune alpha on val, report on test)
#
# KEY FIX vs original:
#   - Original swept alpha on the same data used for reporting.
#     This picks an alpha that overfits to test noise.
#   - This version sweeps alpha on the VAL split, locks the best alpha,
#     then reports AUC/EER on the TEST split only.
#
# Run align_scores.py twice first:
#   python align_scores.py  (SPLIT="val"  → aligned/val/)
#   python align_scores.py  (SPLIT="test" → aligned/test/)
#
# Outputs (in OUTPUT_DIR):
#   weighted_fusion_results.json
#   weighted_auc_curve.png          (alpha sweep on val)
#   weighted_best_test_scores.npy   (fused scores at best alpha, test split)
# =============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
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
    r"\Fusion_Layer\results\weighted"
)

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

    print(f"\n{'='*55}")
    print(f" {tag}")
    print(f"{'='*55}")
    print(f"  AUC                    : {auc:.4f}")
    print(f"  EER                    : {eer:.4f}")
    print(f"  Optimal Threshold      : {opt_thr:.4f}")
    print(f"  Accuracy @ Optimal Thr : {acc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")
    return float(auc), float(eer)


def per_attack_aucs(labels, scores, meta):
    real_l, real_s = [], []
    atk_bkt = defaultdict(lambda: {"l": [], "s": []})
    for i, m in enumerate(meta):
        if labels[i] == 0:
            real_l.append(labels[i]); real_s.append(scores[i])
        else:
            atk_bkt[m["attack"]]["l"].append(labels[i])
            atk_bkt[m["attack"]]["s"].append(scores[i])
    real_l = np.array(real_l); real_s = np.array(real_s)
    result = {}
    for atk, bkt in atk_bkt.items():
        l = np.concatenate([real_l, bkt["l"]])
        s = np.concatenate([real_s, bkt["s"]])
        result[atk] = float(roc_auc_score(l, s))
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load val split
    # -------------------------------------------------------------------------
    print("[INFO] Loading VAL aligned split ...")
    val_vid = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_video_scores.npy"))
    val_aud = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_audio_scores.npy"))
    val_lbl = np.load(os.path.join(VAL_ALIGNED_DIR, "aligned_labels.npy"))
    print(f"  Val  — real: {(val_lbl==0).sum()}  fake: {(val_lbl==1).sum()}  "
          f"total: {len(val_lbl)}")

    # -------------------------------------------------------------------------
    # Load test split
    # -------------------------------------------------------------------------
    print("[INFO] Loading TEST aligned split ...")
    tst_vid = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_video_scores.npy"))
    tst_aud = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_audio_scores.npy"))
    tst_lbl = np.load(os.path.join(TEST_ALIGNED_DIR, "aligned_labels.npy"))
    with open(os.path.join(TEST_ALIGNED_DIR, "aligned_meta.json")) as f:
        tst_meta = json.load(f)
    print(f"  Test — real: {(tst_lbl==0).sum()}  fake: {(tst_lbl==1).sum()}  "
          f"total: {len(tst_lbl)}")

    # -------------------------------------------------------------------------
    # Individual baselines on TEST
    # -------------------------------------------------------------------------
    print("\n========== Baselines (TEST split) ==========")
    aud_auc, aud_eer = print_full_metrics("Audio Only — TEST", tst_lbl, tst_aud)
    vid_auc, vid_eer = print_full_metrics("Video Only — TEST", tst_lbl, tst_vid)

    # -------------------------------------------------------------------------
    # Alpha sweep on VAL split
    # -------------------------------------------------------------------------
    print("\n[INFO] Sweeping alpha on VAL split (alpha = audio weight) ...")
    alphas  = np.round(np.arange(0.0, 1.05, 0.05), 2)
    val_sweep = []

    for alpha in alphas:
        fused = alpha * val_aud + (1 - alpha) * val_vid
        auc   = roc_auc_score(val_lbl, fused)
        eer   = compute_eer(val_lbl, fused)
        val_sweep.append({"alpha": float(alpha), "auc": float(auc), "eer": float(eer)})
        print(f"  alpha={alpha:.2f}  val_AUC={auc:.4f}  val_EER={eer:.4f}")

    best_val = max(val_sweep, key=lambda x: x["auc"])
    best_alpha = best_val["alpha"]
    print(f"\n[INFO] Best alpha on VAL: {best_alpha:.2f}  "
          f"(val AUC={best_val['auc']:.4f})")

    # -------------------------------------------------------------------------
    # Apply best alpha to TEST split (locked — no more tuning)
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Applying best alpha={best_alpha:.2f} to TEST split ...")
    tst_fused = best_alpha * tst_aud + (1 - best_alpha) * tst_vid

    fused_auc, fused_eer = print_full_metrics(
        f"Weighted Fusion α={best_alpha:.2f} — TEST", tst_lbl, tst_fused)

    # -------------------------------------------------------------------------
    # Per-attack on TEST
    # -------------------------------------------------------------------------
    print(f"\n========== Per-Attack (Weighted Fusion — TEST split) ==========")
    atk_results = per_attack_aucs(tst_lbl, tst_fused, tst_meta)
    for atk in sorted(atk_results):
        print(f"  [{atk:<25}]  AUC: {atk_results[atk]:.4f}")

    # -------------------------------------------------------------------------
    # Plot: alpha curve (val AUC) with test baselines for reference
    # -------------------------------------------------------------------------
    val_aucs   = [r["auc"]   for r in val_sweep]
    alpha_vals = [r["alpha"] for r in val_sweep]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha_vals, val_aucs, "b-o", linewidth=2, markersize=6,
            label="Val AUC (sweep)")
    ax.axhline(aud_auc, color="orange", linestyle="--", linewidth=1.5,
               label=f"Audio Only test AUC ({aud_auc:.4f})")
    ax.axhline(vid_auc, color="green",  linestyle="--", linewidth=1.5,
               label=f"Video Only test AUC ({vid_auc:.4f})")
    ax.axvline(best_alpha, color="red", linestyle=":", linewidth=1.5,
               label=f"Best α={best_alpha:.2f} (val)")
    ax.scatter([best_alpha], [best_val["auc"]], color="red", zorder=5, s=100,
               label=f"Best val AUC ({best_val['auc']:.4f})")
    ax.set_xlabel("α  (weight of audio score,  1−α for video)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Weighted Fusion: Val AUC vs Alpha\n(best alpha locked, applied to test)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "weighted_auc_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Saved alpha curve → {plot_path}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "weighted_best_test_scores.npy"), tst_fused)

    summary = {
        "note": "alpha tuned on VAL split; all AUC/EER reported on TEST split.",
        "best_alpha":       best_alpha,
        "val_auc_at_best":  best_val["auc"],
        "val_eer_at_best":  best_val["eer"],
        "audio_only_test":  {"auc": aud_auc, "eer": aud_eer},
        "video_only_test":  {"auc": vid_auc, "eer": vid_eer},
        "fused_test":       {"auc": fused_auc, "eer": fused_eer},
        "per_attack_test":  atk_results,
        "val_sweep":        val_sweep,
    }
    with open(os.path.join(OUTPUT_DIR, "weighted_fusion_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved results → {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "="*55)
    print(" FINAL SUMMARY (TEST split — alpha tuned on val)")
    print("="*55)
    print(f"  {'Method':<35} {'AUC':>8}  {'EER':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Audio Only':<35} {aud_auc:>8.4f}  {aud_eer:>8.4f}")
    print(f"  {'Video Only':<35} {vid_auc:>8.4f}  {vid_eer:>8.4f}")
    print(f"  {f'Weighted Fusion (α={best_alpha:.2f})':<35} "
          f"{fused_auc:>8.4f}  {fused_eer:>8.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
