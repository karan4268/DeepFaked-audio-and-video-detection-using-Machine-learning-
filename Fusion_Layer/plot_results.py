# =============================================================================
# plot_results.py
#
# Generates thesis-quality figures from fusion results:
#   1. Bar chart: AUC comparison across all methods
#   2. Bar chart: Per-attack AUC (audio / video / best fusion)
#   3. AUC vs Alpha curve  (from weighted fusion)
#   4. ROC curves          (from learned fusion)
#   5. Per-speaker heatmap: video AUC vs audio AUC scatter
#
# Run after fuse_weighted.py and fuse_learned.py.
# =============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve

# =============================================================================
# PATHS
# =============================================================================

ALIGNED_DIR      = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\fusion\aligned"
)
WEIGHTED_DIR     = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\fusion\results\weighted"
)
LEARNED_DIR      = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\fusion\results\learned"
)

# Video + Audio per-speaker results (paste from your eval logs or load from saved files)
# These are loaded from the aligned scores + meta for consistency.
OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\fusion\results\plots"
)

# =============================================================================


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def set_style():
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":     11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":    150,
    })


def plot_auc_bar(methods, aucs, eers, output_path):
    """Figure 1: Overall AUC bar chart with EER annotations."""
    colors = ["#F4A261", "#2A9D8F", "#457B9D", "#E63946"]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, aucs, color=colors[:len(methods)], width=0.5, zorder=3)

    # Value labels
    for bar, auc, eer in zip(bars, aucs, eers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"AUC={auc:.4f}\nEER={eer:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("AUC")
    ax.set_title("Deepfake Detection: Overall AUC Comparison")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {output_path}")


def plot_per_attack_bar(attacks, aud_aucs, vid_aucs, fused_aucs, output_path):
    """Figure 2: Grouped bar chart per attack."""
    x      = np.arange(len(attacks))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width, aud_aucs,   width, label="Audio Only",    color="#F4A261")
    b2 = ax.bar(x,          vid_aucs,   width, label="Video Only",    color="#2A9D8F")
    b3 = ax.bar(x + width,  fused_aucs, width, label="Best Fusion",   color="#457B9D")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10)
    ax.set_ylabel("AUC")
    ax.set_title("Per-Attack AUC: Audio vs Video vs Fusion")
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {output_path}")


def plot_alpha_curve(weighted_results_path, output_path):
    """Figure 3: AUC vs Alpha from weighted fusion."""
    with open(weighted_results_path) as f:
        res = json.load(f)

    alphas = [r["alpha"] for r in res["sweep"]]
    aucs   = [r["auc"]   for r in res["sweep"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, aucs, "b-o", linewidth=2, markersize=5, label="Fusion AUC")
    ax.axhline(res["audio_only_auc"], color="#F4A261", linestyle="--", linewidth=1.8,
               label=f"Audio Only ({res['audio_only_auc']:.4f})")
    ax.axhline(res["video_only_auc"], color="#2A9D8F", linestyle="--", linewidth=1.8,
               label=f"Video Only ({res['video_only_auc']:.4f})")
    ax.axvline(res["best_alpha"], color="red", linestyle=":", linewidth=1.5,
               label=f"Best α={res['best_alpha']:.2f}  AUC={res['best_auc']:.4f}")
    ax.scatter([res["best_alpha"]], [res["best_auc"]], color="red", zorder=5, s=80)
    ax.set_xlabel("α  (weight of audio score,  1−α for video)")
    ax.set_ylabel("AUC")
    ax.set_title("Weighted Fusion: AUC vs Audio Weight (α)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {output_path}")


def plot_roc_curves(labels, aud_scores, vid_scores, lr_scores, mlp_scores, output_path):
    """Figure 4: ROC curves for all methods."""
    fig, ax = plt.subplots(figsize=(8, 7))

    configs = [
        (aud_scores, "Audio Only",    "#F4A261", "--"),
        (vid_scores, "Video Only",    "#2A9D8F", "--"),
        (lr_scores,  "LR Fusion",     "#457B9D", "-"),
        (mlp_scores, "MLP Fusion",    "#E63946", "-"),
    ]

    for scores, label, color, ls in configs:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{label}  (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Individual Models vs Fusion")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {output_path}")


def plot_speaker_scatter(labels, aud_scores, vid_scores, meta, output_path):
    """Figure 5: Per-speaker scatter — video AUC vs audio AUC."""
    from collections import defaultdict

    real_pool_l, real_pool_va, real_pool_aa = [], [], []
    spk_buckets = defaultdict(lambda: {"labels": [], "aud": [], "vid": []})

    for i, m in enumerate(meta):
        if labels[i] == 0:
            real_pool_l.append(labels[i])
            real_pool_va.append(vid_scores[i])
            real_pool_aa.append(aud_scores[i])
        else:
            spk = m["speaker"]
            spk_buckets[spk]["labels"].append(labels[i])
            spk_buckets[spk]["aud"].append(aud_scores[i])
            spk_buckets[spk]["vid"].append(vid_scores[i])

    real_l  = np.array(real_pool_l)
    real_va = np.array(real_pool_va)
    real_aa = np.array(real_pool_aa)

    spk_vid_aucs, spk_aud_aucs = [], []

    for spk, bkt in spk_buckets.items():
        l   = np.concatenate([real_l,  bkt["labels"]])
        va  = np.concatenate([real_va, bkt["vid"]])
        aa  = np.concatenate([real_aa, bkt["aud"]])
        if len(np.unique(l)) < 2:
            continue
        spk_vid_aucs.append(roc_auc_score(l, va))
        spk_aud_aucs.append(roc_auc_score(l, aa))

    spk_vid_aucs = np.array(spk_vid_aucs)
    spk_aud_aucs = np.array(spk_aud_aucs)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(spk_vid_aucs, spk_aud_aucs, alpha=0.6, s=40,
                    c=spk_aud_aucs - spk_vid_aucs, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Equal performance")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Audio AUC − Video AUC")
    ax.set_xlabel("Video Model AUC (per speaker)")
    ax.set_ylabel("Audio Model AUC (per speaker)")
    ax.set_title("Per-Speaker: Audio vs Video AUC\n(green=audio better, red=video better)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {output_path}")


def main():
    set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load everything
    # -------------------------------------------------------------------------
    vid_scores = np.load(os.path.join(ALIGNED_DIR, "aligned_video_scores.npy"))
    aud_scores = np.load(os.path.join(ALIGNED_DIR, "aligned_audio_scores.npy"))
    labels     = np.load(os.path.join(ALIGNED_DIR, "aligned_labels.npy"))
    lr_scores  = np.load(os.path.join(LEARNED_DIR,  "lr_fusion_scores.npy"))
    mlp_scores = np.load(os.path.join(LEARNED_DIR,  "mlp_fusion_scores.npy"))
    best_w_scores = np.load(os.path.join(WEIGHTED_DIR, "weighted_best_scores.npy"))

    with open(os.path.join(ALIGNED_DIR,  "aligned_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(WEIGHTED_DIR, "weighted_fusion_results.json")) as f:
        w_res = json.load(f)
    with open(os.path.join(LEARNED_DIR,  "learned_fusion_results.json")) as f:
        l_res = json.load(f)

    # -------------------------------------------------------------------------
    # Figure 1: Overall AUC bar chart
    # -------------------------------------------------------------------------
    methods = ["Audio Only", "Video Only", "LR Fusion", "MLP Fusion"]
    aucs    = [
        l_res["audio_only"]["auc"],
        l_res["video_only"]["auc"],
        l_res["lr_fusion"]["auc"],
        l_res["mlp_fusion"]["auc"],
    ]
    eers    = [
        l_res["audio_only"]["eer"],
        l_res["video_only"]["eer"],
        l_res["lr_fusion"]["eer"],
        l_res["mlp_fusion"]["eer"],
    ]
    plot_auc_bar(methods, aucs, eers,
                 os.path.join(OUTPUT_DIR, "fig1_overall_auc.png"))

    # -------------------------------------------------------------------------
    # Figure 2: Per-attack AUC
    # -------------------------------------------------------------------------
    from collections import defaultdict

    def per_attack_aucs(scores):
        real_l, real_s = [], []
        atk_bkt = defaultdict(lambda: {"labels": [], "scores": []})
        for i, m in enumerate(meta):
            if labels[i] == 0:
                real_l.append(labels[i]); real_s.append(scores[i])
            else:
                atk_bkt[m["attack"]]["labels"].append(labels[i])
                atk_bkt[m["attack"]]["scores"].append(scores[i])
        real_l = np.array(real_l); real_s = np.array(real_s)
        result = {}
        for atk, bkt in atk_bkt.items():
            l = np.concatenate([real_l, bkt["labels"]])
            s = np.concatenate([real_s, bkt["scores"]])
            result[atk] = roc_auc_score(l, s)
        return result

    attacks_order = ["faceswap-wav2lip", "fsgan-wav2lip", "rtvc", "wav2lip"]
    aud_atk = per_attack_aucs(aud_scores)
    vid_atk = per_attack_aucs(vid_scores)
    lr_atk  = per_attack_aucs(lr_scores)

    # Use friendly names for plot
    atk_labels = ["FaceSwap-Wav2Lip", "FSGAN-Wav2Lip", "RTVC", "Wav2Lip"]
    plot_per_attack_bar(
        atk_labels,
        [aud_atk.get(a, 0) for a in attacks_order],
        [vid_atk.get(a, 0) for a in attacks_order],
        [lr_atk.get(a,  0) for a in attacks_order],
        os.path.join(OUTPUT_DIR, "fig2_per_attack_auc.png")
    )

    # -------------------------------------------------------------------------
    # Figure 3: Alpha curve
    # -------------------------------------------------------------------------
    plot_alpha_curve(
        os.path.join(WEIGHTED_DIR, "weighted_fusion_results.json"),
        os.path.join(OUTPUT_DIR, "fig3_alpha_curve.png")
    )

    # -------------------------------------------------------------------------
    # Figure 4: ROC curves
    # -------------------------------------------------------------------------
    plot_roc_curves(
        labels, aud_scores, vid_scores, lr_scores, mlp_scores,
        os.path.join(OUTPUT_DIR, "fig4_roc_curves.png")
    )

    # -------------------------------------------------------------------------
    # Figure 5: Per-speaker scatter
    # -------------------------------------------------------------------------
    plot_speaker_scatter(
        labels, aud_scores, vid_scores, meta,
        os.path.join(OUTPUT_DIR, "fig5_speaker_scatter.png")
    )

    print(f"\n[INFO] All figures saved → {OUTPUT_DIR}")
    print("  fig1_overall_auc.png")
    print("  fig2_per_attack_auc.png")
    print("  fig3_alpha_curve.png")
    print("  fig4_roc_curves.png")
    print("  fig5_speaker_scatter.png")


if __name__ == "__main__":
    main()
