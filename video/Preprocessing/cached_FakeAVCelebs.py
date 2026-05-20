# =============================================================================
# cache_fakeavceleb_video.py (FIXED — fusion-aligned index)
# =============================================================================
import sys
import os

ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
sys.path.insert(0, ROOT)

import cv2
import json
import torch
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor

from Fusion_Layer.fusion_utility import make_file_id


# =============================================================================
# CONFIG
# =============================================================================

FAKEAV_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"
META_CSV    = os.path.join(FAKEAV_ROOT, "meta_data.csv")

CACHE_ROOT  = r"D:\FakeAVCache\Video"

FRAMES_PER_VIDEO = 24
IMG_SIZE         = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# DETECTOR
# =============================================================================

detector = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=True,
    device=DEVICE
)


# =============================================================================
# CACHE PATH
# FIX: "file" is now a CACHE-ROOT-RELATIVE path "{hid}/frames.npy"
#      so fusion loader can resolve: os.path.join(VIDEO_CACHE_ROOT, entry["file"])
#      This matches the audio convention of storing a resolvable relative path.
# =============================================================================

def get_file_id(path):
    return make_file_id(path, FAKEAV_ROOT)


def cache_path(video_path):
    file_id = get_file_id(video_path)

    signature = f"fakeav|{file_id}"
    hid       = hashlib.md5(signature.encode()).hexdigest()

    # Relative path from CACHE_ROOT — used directly in the index
    rel_path  = os.path.join(hid, "frames.npy")
    out_dir   = os.path.join(CACHE_ROOT, hid)
    out_npy   = os.path.join(CACHE_ROOT, rel_path)

    return out_dir, out_npy, rel_path, file_id


# =============================================================================
# FRAME SAMPLING
# =============================================================================

def sample_frame_ids(total):
    return np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int).tolist()


def center_crop(frame):
    h, w, _ = frame.shape
    m = min(h, w)
    y = (h - m) // 2
    x = (w - m) // 2
    frame = frame[y:y+m, x:x+m]
    return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))


# =============================================================================
# PROCESS VIDEO
# =============================================================================

def process_video(entry):
    try:
        path = entry["full_path"]

        if not os.path.isfile(path):
            return {"skip": "missing"}

        out_dir, out_npy, rel_path, file_id = cache_path(path)

        # Cache hit
        if os.path.isfile(out_npy):
            # -----------------------------------------------------------------
            # FIX 1: "file" is now "{hid}/frames.npy" — CACHE-ROOT-RELATIVE
            #         → fusion loader: np.load(os.path.join(VIDEO_CACHE_ROOT, entry["file"]))
            # FIX 2: "modality" field added for explicit fusion dispatch
            # -----------------------------------------------------------------
            return {
                "file_id":  file_id,
                "file":     rel_path,          # e.g. "abc123/frames.npy"
                "label":    entry["label"],
                "speaker":  entry["speaker"],
                "attack":   entry["method"] if entry["label"] == 1 else "bonafide",
                "dataset":  "fakeav",
                "modality": "video",
            }

        os.makedirs(out_dir, exist_ok=True)

        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return {"skip": "bad_video"}

        frame_ids = set(sample_frame_ids(total))
        frames    = []
        idx       = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_ids:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == FRAMES_PER_VIDEO:
                    break
            idx += 1

        cap.release()

        if len(frames) == 0:
            return {"skip": "no_frames"}

        # Face processing
        final_frames = []
        for f in frames:
            boxes, probs = detector.detect(Image.fromarray(f))

            if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                h, w, _         = f.shape
                x1, y1          = max(0, x1), max(0, y1)
                x2, y2          = min(w, x2), min(h, y2)
                crop            = f[y1:y2, x1:x2]
                face            = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) if crop.size > 0 else center_crop(f)
            else:
                face = center_crop(f)

            final_frames.append(face)

        if len(final_frames) != FRAMES_PER_VIDEO:
            return {"skip": "incomplete_frames"}

        clip = np.stack(final_frames, axis=0).astype(np.uint8)
        np.save(out_npy, clip)

        return {
            "file_id":  file_id,
            "file":     rel_path,          # e.g. "abc123/frames.npy"
            "label":    entry["label"],
            "speaker":  entry["speaker"],
            "attack":   entry["method"] if entry["label"] == 1 else "bonafide",
            "dataset":  "fakeav",
            "modality": "video",
        }

    except Exception as e:
        return {"skip": str(e)}


# =============================================================================
# BUILD
# =============================================================================

def build_cache():
    print("\n===== FakeAVCeleb Video Cache (fusion-aligned) =====")

    df = pd.read_csv(META_CSV)
    df.columns = [c.strip() for c in df.columns]

    df["label"] = df["category"].apply(lambda c: 0 if c in ["A", "C"] else 1)

    dir_col = None
    for c in df.columns:
        if c.lower() not in ["source", "target1", "target2", "method",
                              "category", "type", "race", "gender", "path"]:
            dir_col = c
            break

    if dir_col is None:
        raise ValueError("Directory column not found in metadata CSV")

    entries = []
    for _, r in df.iterrows():
        filename  = str(r["path"]).strip()
        directory = str(r[dir_col]).strip()

        if directory.startswith("FakeAVCeleb"):
            directory = directory.replace("FakeAVCeleb", "").lstrip("\\/")

        full_path = os.path.join(FAKEAV_ROOT, directory, filename)

        entries.append({
            "full_path": full_path,
            "label":     int(r["label"]),
            "speaker":   r.get("source", "unknown"),
            "method":    r.get("method", "unknown"),
        })

    print(f"[INFO] Videos: {len(entries)}")

    index = []

    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in tqdm(ex.map(process_video, entries), total=len(entries)):
            if "skip" not in r:
                index.append(r)

    os.makedirs(CACHE_ROOT, exist_ok=True)

    save_path = os.path.join(CACHE_ROOT, "video_index.json")
    with open(save_path, "w") as f:
        json.dump(index, f, indent=2)

    real = sum(1 for e in index if e["label"] == 0)
    fake = sum(1 for e in index if e["label"] == 1)

    print("\n===== DONE =====")
    print(f"Cached: {len(index)}  (real={real}, fake={fake})")
    print(f"[INDEX] {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    build_cache()
    print("\n✅ FakeAVCeleb video cache complete.")