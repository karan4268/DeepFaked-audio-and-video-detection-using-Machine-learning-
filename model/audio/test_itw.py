# =============================================================================
# test_itw.py (AUTO MODE: RAW + CACHE)
# =============================================================================

import os
import argparse
import csv
import torch
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve

from audio_model import AudioResNet18
from Preprocessing.extractor import compute_log_mel


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\audio_combined_best.pth"
)

DEFAULT_OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Models\audio\eval_outputs_itw"
)

DEFAULT_ITW_ROOT = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Data\Audio\cache_wave\itw_test"
)


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_FRAMES  = 400
SAMPLE_RATE = 16000

RAW_EXTENSIONS   = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
CACHE_EXTENSIONS = {".npy"}


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def print_metrics(tag, labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class present — skipping (N={len(labels)})")
        return

    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)

    print(f"  [{tag}] AUC: {auc:.4f}  EER: {eer:.4f}  N={len(labels)}")


# =============================================================================
# DATASET (AUTO MODE)
# =============================================================================

class ITWAudioDataset(Dataset):

    def __init__(self, itw_root):
        self.samples = []
        self.mode = None  # "raw" or "cache"

        real_dir = os.path.join(itw_root, "real")
        fake_dir = os.path.join(itw_root, "fake")

        # Detect mode from first valid file
        def detect_mode(file):
            ext = os.path.splitext(file)[1].lower()
            if ext in RAW_EXTENSIONS:
                return "raw"
            if ext in CACHE_EXTENSIONS:
                return "cache"
            return None

        # Scan labelled layout
        if os.path.isdir(real_dir) and os.path.isdir(fake_dir):

            for label_dir, label in [(real_dir, 0), (fake_dir, 1)]:
                for f in sorted(os.listdir(label_dir)):
                    ext = os.path.splitext(f)[1].lower()

                    if ext in RAW_EXTENSIONS or ext in CACHE_EXTENSIONS:
                        if self.mode is None:
                            self.mode = detect_mode(f)

                        self.samples.append((os.path.join(label_dir, f), label))

            n_real = sum(1 for _, l in self.samples if l == 0)
            n_fake = sum(1 for _, l in self.samples if l == 1)

            print(f"[INFO] Labelled layout — real: {n_real}  fake: {n_fake}")

        else:
            # Flat layout
            for f in sorted(os.listdir(itw_root)):
                full = os.path.join(itw_root, f)

                if not os.path.isfile(full):
                    continue

                ext = os.path.splitext(f)[1].lower()

                if ext in RAW_EXTENSIONS or ext in CACHE_EXTENSIONS:
                    if self.mode is None:
                        self.mode = detect_mode(f)

                    self.samples.append((full, -1))

            print(f"[INFO] Flat layout — {len(self.samples)} unlabelled files")

        if self.mode is None:
            raise RuntimeError("No supported files found (.wav or .npy)")

        print(f"[INFO] Detected mode: {self.mode.upper()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ============================================================
        # RAW MODE
        # ============================================================
        if self.mode == "raw":
            try:
                audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"[WARN] Cannot load {path}: {e} — zero-filling")
                audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

            audio = np.nan_to_num(audio.astype(np.float32))
            audio = np.clip(audio, -1.0, 1.0)

            log_mel = compute_log_mel(audio)

        # ============================================================
        # CACHE MODE (WAVEFORM CACHE — CORRECT HANDLING)
        # ============================================================
        try:
            audio = np.load(path)
        except Exception as e:
            print(f"[WARN] Cannot load {path}: {e} — zero-filling")
            audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

        # Convert int16 → float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        else:
            audio = audio.astype(np.float32)

        # Safety
        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)

        # NOW compute features (same as training)
        log_mel = compute_log_mel(audio)

        # ============================================================
        # COMMON POST-PROCESSING
        # ============================================================
        log_mel = np.nan_to_num(log_mel)

        T = log_mel.shape[-1]

        if T > MAX_FRAMES:
            start = (T - MAX_FRAMES) // 2
            log_mel = log_mel[:, start:start + MAX_FRAMES]
        elif T < MAX_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - T)))

        x = torch.from_numpy(log_mel).unsqueeze(0).float()

        return x, torch.tensor(label, dtype=torch.float32), path


def itw_collate(batch):
    xs    = torch.stack([b[0] for b in batch])
    ys    = torch.stack([b[1] for b in batch])
    paths = [b[2] for b in batch]
    return xs, ys, paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ITW evaluation (AUTO MODE)")
    parser.add_argument("--itw_root", default=DEFAULT_ITW_ROOT)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Using ITW root: {args.itw_root}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = AudioResNet18().to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[INFO] Loaded checkpoint")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset = ITWAudioDataset(args.itw_root)

    if len(dataset) == 0:
        print("[ERROR] No valid files found")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=itw_collate
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    all_labels = []
    all_scores = []
    all_paths  = []

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)

            logits = model(x).view(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(probs)
            all_labels.extend(y.numpy())
            all_paths.extend(paths)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    print("\n========== ITW Results ==========")

    if (all_labels >= 0).all():
        print_metrics("ITW", all_labels, all_scores)
    else:
        print(f"{'File':<50} {'Score':>7}  Verdict")
        print(f"{'-'*50} {'-'*7}  -------")

        for path, score in zip(all_paths, all_scores):
            verdict = "FAKE" if score >= 0.5 else "REAL"
            print(f"{os.path.basename(path):<50} {score:>7.4f}  {verdict}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "itw_predictions.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "score", "pred"])

        for p, l, s in zip(all_paths, all_labels, all_scores):
            writer.writerow([p, int(l), float(s), int(s >= 0.5)])

    np.save(os.path.join(args.output_dir, "itw_scores.npy"), all_scores)
    np.save(os.path.join(args.output_dir, "itw_labels.npy"), all_labels)

    print(f"\n[INFO] Saved outputs → {args.output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()