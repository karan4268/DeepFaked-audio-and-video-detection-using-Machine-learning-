# =============================================================================
# Real-world inference script for Audio-Visual Deepfake Detection.
# Runs video model + audio model on raw MP4 files and fuses scores.
#
# Usage:
#   Single file:
#     final test.py --input clip.mp4
#
#   Batch folder:
#     final test.py --input path/to/folder
#
#   Optional flags:
#     --alpha 0.45              Override fusion alpha (default: 0.45)
#     --threshold 0.5           Decision threshold (default: 0.5)
#     --output results.json     Save results to JSON (default: print only)
#     --device cuda             Force device (default: auto)
#
# Requirements (beyond your existing project deps):
#   pip install ffmpeg-python   (or just have ffmpeg on PATH)
#   pip install opencv-python
#
# The script extracts frames + audio from MP4 in-memory (no cache files).
# =============================================================================

import os
import sys
import json
import argparse
import tempfile
import subprocess
import numpy as np
import torch

from video.video_model import VideoModel
from audio.audio_model import AudioResNet18
from audio.Preprocessing.extractor import compute_log_mel

# =============================================================================
# PATHS — edit these
# =============================================================================

VIDEO_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video_finetuned\finetuned_best.pth"
)

AUDIO_MODEL_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\audio_combined_best.pth"
)

# Best alpha from val sweep — override with --alpha flag
DEFAULT_ALPHA     = 0.45   # audio weight
DEFAULT_THRESHOLD = 0.55    # above = FAKE
NUM_FRAMES        = 24     # frames sampled per clip (must match training)
AUDIO_SR          = 16000  # sample rate expected by audio model
MAX_MEL_FRAMES    = 400    # must match training

# ImageNet normalization (must match video training)
VID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
VID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# =============================================================================
# VIDEO PREPROCESSING
# Extracts NUM_FRAMES evenly spaced frames from MP4 using OpenCV.
# Returns tensor of shape (1, C, T, H, W) ready for VideoModel.
# =============================================================================

def extract_video_frames(mp4_path: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required: pip install opencv-python")

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {mp4_path}")

    # Evenly spaced indices (same logic as ValSplitDataset)
    if total >= num_frames:
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        idxs = np.concatenate([
            np.arange(total),
            np.full(num_frames - total, total - 1)
        ])

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Use last good frame if seek fails
            frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    clip = np.transpose(clip, (0, 3, 1, 2))                      # (T, C, H, W)
    clip = (clip - VID_MEAN) / VID_STD
    clip = torch.from_numpy(clip).permute(1, 0, 2, 3)            # (C, T, H, W)
    return clip.unsqueeze(0)                                      # (1, C, T, H, W)


# =============================================================================
# AUDIO PREPROCESSING
# Extracts mono 16kHz audio from MP4 using ffmpeg subprocess.
# Returns log-mel tensor of shape (1, 1, 80, MAX_MEL_FRAMES).
# =============================================================================

def extract_audio_logmel(mp4_path: str) -> torch.Tensor:
    # Use ffmpeg to extract raw PCM s16le at 16kHz mono into a temp file
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-vn",                        # no video
            "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_SR),
            "-ac", "1",                   # mono
            "-f", "s16le",
            tmp_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            # If no audio stream, return a silent (zero) mel
            if "no audio" in err.lower() or "stream" in err.lower():
                print(f"  [WARN] No audio stream found in {os.path.basename(mp4_path)} — using silence")
                audio = np.zeros(AUDIO_SR, dtype=np.float32)
            else:
                raise RuntimeError(f"ffmpeg failed for {mp4_path}:\n{err}")
        else:
            raw = np.fromfile(tmp_path, dtype=np.int16)
            audio = raw.astype(np.float32) / 32768.0  # normalise to [-1, 1]

            if len(audio) == 0:
                print(f"  [WARN] Empty audio in {os.path.basename(mp4_path)} — using silence")
                audio = np.zeros(AUDIO_SR, dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Compute log-mel (same function used during training)
    log_mel = compute_log_mel(audio)
    log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad / crop to MAX_MEL_FRAMES (same as FakeAVAudioDataset)
    T = log_mel.shape[-1]
    if T > MAX_MEL_FRAMES:
        start   = (T - MAX_MEL_FRAMES) // 2
        log_mel = log_mel[:, start:start + MAX_MEL_FRAMES]
    elif T < MAX_MEL_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_MEL_FRAMES - T)))

    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()  # (1,1,80,400)
    return tensor


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_video_model(path: str, device: torch.device) -> torch.nn.Module:
    model = VideoModel(num_classes=2, pretrained=False, dropout=0.3).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_auc = ckpt.get("val_auc", "?")
    print(f"  [VIDEO] Loaded  — checkpoint val_auc: {val_auc}")
    return model


def load_audio_model(path: str, device: torch.device) -> torch.nn.Module:
    model = AudioResNet18().to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [AUDIO] Loaded")
    return model


# =============================================================================
# SINGLE-CLIP INFERENCE
# =============================================================================

def infer_clip(
    mp4_path:   str,
    vid_model:  torch.nn.Module,
    aud_model:  torch.nn.Module,
    device:     torch.device,
    alpha:      float,
    threshold:  float,
    use_amp:    bool,
) -> dict:
    """
    Run both models on one MP4 clip and return a result dict.
    """
    name = os.path.basename(mp4_path)

    # ── Video ────────────────────────────────────────────────────────────────
    try:
        vid_tensor = extract_video_frames(mp4_path).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    vid_logits = vid_model(vid_tensor)
            else:
                vid_logits = vid_model(vid_tensor)

        if vid_logits.ndim == 2:
            vid_score = float(torch.softmax(vid_logits, dim=1)[0, 1].cpu())
        else:
            vid_score = float(torch.sigmoid(vid_logits.view(-1))[0].cpu())
        vid_error = None
    except Exception as e:
        vid_score = 0.5   # neutral if extraction fails
        vid_error = str(e)
        print(f"  [WARN] Video extraction failed for {name}: {e}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    try:
        aud_tensor = extract_audio_logmel(mp4_path).to(device)
        with torch.no_grad():
            aud_logits = aud_model(aud_tensor).view(-1)
            aud_score  = float(torch.sigmoid(aud_logits)[0].cpu())
        aud_error = None
    except Exception as e:
        aud_score = 0.5
        aud_error = str(e)
        print(f"  [WARN] Audio extraction failed for {name}: {e}")

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused_score = alpha * aud_score + (1.0 - alpha) * vid_score
    verdict     = "FAKE" if fused_score >= threshold else "REAL"

    return {
        "file":         mp4_path,
        "verdict":      verdict,
        "fused_score":  round(fused_score, 4),
        "video_score":  round(vid_score,   4),
        "audio_score":  round(aud_score,   4),
        "alpha":        alpha,
        "threshold":    threshold,
        "video_error":  vid_error,
        "audio_error":  aud_error,
    }


# =============================================================================
# PRETTY PRINT
# =============================================================================

def print_result(r: dict):
    bar_len  = 30
    filled   = int(r["fused_score"] * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    verdict  = r["verdict"]
    tag      = "🔴 FAKE" if verdict == "FAKE" else "🟢 REAL"

    print(f"\n  {'─'*52}")
    print(f"  File    : {os.path.basename(r['file'])}")
    print(f"  Verdict : {tag}")
    print(f"  {'─'*52}")
    print(f"  Video   score : {r['video_score']:.4f}")
    print(f"  Audio   score : {r['audio_score']:.4f}")
    print(f"  Fused   score : {r['fused_score']:.4f}  [{bar}]")
    print(f"  Alpha={r['alpha']:.2f}  |  Threshold={r['threshold']:.2f}")
    if r["video_error"]:
        print(f"  ⚠ Video error : {r['video_error']}")
    if r["audio_error"]:
        print(f"  ⚠ Audio error : {r['audio_error']}")
    print(f"  {'─'*52}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Audio-Visual Deepfake Detector — real-world inference"
    )
    # add path argument for single MP4 or folder of MP4s here or though terminal
    p.add_argument(
        "--input", "-i", default="F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\My test clips",
        help="Path to a single MP4 file OR a folder of MP4 files"
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Audio fusion weight (default: {DEFAULT_ALPHA}). "
             f"video weight = 1 - alpha"
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Fused score threshold for FAKE verdict (default: {DEFAULT_THRESHOLD})"
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Optional path to save results as JSON (e.g. results.json)"
    )
    p.add_argument(
        "--device", default=None,
        help="Force device: 'cuda' or 'cpu' (default: auto)"
    )
    p.add_argument(
        "--video_model", default=VIDEO_MODEL_PATH,
        help="Override path to video model checkpoint"
    )
    p.add_argument(
        "--audio_model", default=AUDIO_MODEL_PATH,
        help="Override path to audio model checkpoint"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    dev_name = torch.cuda.get_device_name(device) if use_amp else "CPU"
    print(f"\n[INFO] Device : {device} | {dev_name}")
    print(f"[INFO] Alpha  : {args.alpha:.2f}  (audio={args.alpha:.2f}, video={1-args.alpha:.2f})")
    print(f"[INFO] Threshold : {args.threshold:.2f}")

    # ── Models ────────────────────────────────────────────────────────────────
    print("\n[INFO] Loading models...")
    vid_model = load_video_model(args.video_model, device)
    aud_model = load_audio_model(args.audio_model, device)

    # ── Collect input files ───────────────────────────────────────────────────
    inp = args.input
    if os.path.isfile(inp):
        if not inp.lower().endswith(".mp4"):
            print(f"[ERROR] Input file must be an MP4: {inp}")
            sys.exit(1)
        mp4_files = [inp]
        print(f"\n[INFO] Single file mode: {inp}")
    elif os.path.isdir(inp):
        mp4_files = sorted([
            os.path.join(inp, f)
            for f in os.listdir(inp)
            if f.lower().endswith(".mp4")
        ])
        if not mp4_files:
            print(f"[ERROR] No MP4 files found in folder: {inp}")
            sys.exit(1)
        print(f"\n[INFO] Batch mode: {len(mp4_files)} MP4 files in {inp}")
    else:
        print(f"[ERROR] Input path not found: {inp}")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []
    fake_count = 0
    real_count = 0

    for i, mp4_path in enumerate(mp4_files, 1):
        print(f"\n[{i}/{len(mp4_files)}] Processing: {os.path.basename(mp4_path)}")
        r = infer_clip(
            mp4_path  = mp4_path,
            vid_model = vid_model,
            aud_model = aud_model,
            device    = device,
            alpha     = args.alpha,
            threshold = args.threshold,
            use_amp   = use_amp,
        )
        print_result(r)
        results.append(r)

        if r["verdict"] == "FAKE":
            fake_count += 1
        else:
            real_count += 1

    # ── Batch summary ─────────────────────────────────────────────────────────
    if len(results) > 1:
        avg_vid   = np.mean([r["video_score"] for r in results])
        avg_aud   = np.mean([r["audio_score"] for r in results])
        avg_fused = np.mean([r["fused_score"] for r in results])

        print(f"\n{'='*54}")
        print(f"  BATCH SUMMARY  ({len(results)} clips)")
        print(f"{'='*54}")
        print(f"  🟢 REAL : {real_count}   🔴 FAKE : {fake_count}")
        print(f"  Avg video score : {avg_vid:.4f}")
        print(f"  Avg audio score : {avg_aud:.4f}")
        print(f"  Avg fused score : {avg_fused:.4f}")
        print(f"{'='*54}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        out = {
            "summary": {
                "total":      len(results),
                "real":       real_count,
                "fake":       fake_count,
                "alpha":      args.alpha,
                "threshold":  args.threshold,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[INFO] Results saved → {args.output}")


if __name__ == "__main__":
    main()