# ------------------------------------------------------------ 
# MTech Project: DeepFaked Video and Audio Analyzer # Module: Audio Preprocessing - cache_audio.py 
# Uses VIDEO dataset_split.json as master split # Creates audio cache aligned with video hash IDs 
# ------------------------------------------------------------ 
# Primary labels (4-class): 
# 0: RealVideo-RealAudio (RR) -> video real, audio real 
# 1: FakeVideo-RealAudio (FR) -> video fake, audio real 
# 2: RealVideo-FakeAudio (RF) -> video real, audio fake 
# 3: FakeVideo-FakeAudio (FF) -> video fake, audio fake 
# ------------------------------------------------------------ 
# Binary labels (for audio-only model): 
# The audio model should detect whether the AUDIO is fake. 
# 0: Real Audio (RR + FR) 
# 1: Fake Audio (RF + FF) 
# ------------------------------------------------------------ 
# AUDIO_BINARY_MAP 
# 0: 0 # RR -> real audio 
# 1: 0 # FR -> real audio (video fake but audio real) 
# 2: 1 # RF -> fake audio 
# 3: 1 # FF -> fake audio 

# label.txt format: 
# primary_label,binary_label 
# ------------------------------------------------------------
# FIXED Audio Cache Script (Aligned with dataset_split_fixed.json)
# ------------------------------------------------------------

import os
import json
import sys
import shutil
import subprocess
import tempfile
import hashlib
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from extractor import compute_log_mel 

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"

# ✅ USE FIXED SPLIT (IMPORTANT)
VIDEO_SPLIT_PATH = os.path.join(ROOT_DIR, "dataset_split_fixed.json")

AUDIO_CACHE = os.path.join(ROOT_DIR, "cached_audio")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

SAMPLE_RATE = 16000

# ------------------------------------------------------------
# AUDIO LABEL MAP
# ------------------------------------------------------------

AUDIO_BINARY_MAP = {
    0: 0,  # RR
    1: 0,  # FR
    2: 1,  # RF
    3: 1   # FF
}

# ------------------------------------------------------------
# CHECK FFMPEG
# ------------------------------------------------------------

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found")
        sys.exit(1)
    print("FFmpeg detected")

# ------------------------------------------------------------
# BUILD VIDEO INDEX (hash → video path)
# ------------------------------------------------------------

def build_video_index():
    print("Building video index...")
    index = {}

    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                video_path = os.path.join(root, f)

                rel = os.path.relpath(video_path, ROOT_DIR)
                hash_id = hashlib.md5(rel.encode("utf-8")).hexdigest()

                index[hash_id] = video_path

    print(f"Indexed {len(index)} videos")
    return index

# ------------------------------------------------------------
# EXTRACT AUDIO (FFMPEG)
# ------------------------------------------------------------

def extract_audio_temp(video_path):
    temp_dir = tempfile.gettempdir()

    wav_path = os.path.join(
        temp_dir,
        f"audio_{hashlib.md5(video_path.encode()).hexdigest()}.wav"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        wav_path
    ]

    subprocess.run(cmd, check=True)

    if not os.path.exists(wav_path):
        raise RuntimeError("FFmpeg failed")

    return wav_path

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def cache_audio_worker(args):
    video_path, hash_id, primary_label = args

    out_dir = os.path.join(AUDIO_CACHE, hash_id)
    os.makedirs(out_dir, exist_ok=True)

    feature_path = os.path.join(out_dir, "logmel.npy")
    label_path = os.path.join(out_dir, "label.txt")

    # ✅ Resume safe
    if os.path.exists(feature_path):
        return {
            "path": out_dir,
            "primary_label": primary_label,
            "binary_label": AUDIO_BINARY_MAP[primary_label]
        }

    wav_path = None

    try:
        wav_path = extract_audio_temp(video_path)

        feat = compute_log_mel(wav_path)
        np.save(feature_path, feat)

        binary = AUDIO_BINARY_MAP[primary_label]

        with open(label_path, "w") as f:
            f.write(f"{primary_label},{binary}")

        return {
            "path": out_dir,
            "primary_label": primary_label,
            "binary_label": binary
        }

    except Exception as e:
        print("\nFAILED:", video_path)
        print("Error:", e)
        return None

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def process_audio():

    with open(VIDEO_SPLIT_PATH) as f:
        split = json.load(f)

    os.makedirs(AUDIO_CACHE, exist_ok=True)

    video_index = build_video_index()

    audio_split = {
        "train": [],
        "val": [],
        "test": []
    }

    print("Processing audio cache...")

    for split_name in ["train", "val", "test"]:

        items = []

        for item in split[split_name]:

            hash_id = os.path.basename(item["path"])
            primary_label = item["primary_label"]

            video_path = video_index.get(hash_id)

            if video_path is None:
                continue

            items.append((video_path, hash_id, primary_label))

        workers = min(8, cpu_count())

        with Pool(workers) as pool:

            for res in tqdm(
                pool.imap_unordered(cache_audio_worker, items),
                total=len(items),
                desc=split_name
            ):
                if res:
                    audio_split[split_name].append(res)

    # ✅ SAVE FIXED AUDIO SPLIT
    out_json = os.path.join(ROOT_DIR, "audio_dataset_split_fixed.json")

    with open(out_json, "w") as f:
        json.dump(audio_split, f, indent=2)

    print("\n✅ Audio cache aligned with fixed split created")

# ------------------------------------------------------------

if __name__ == "__main__":
    check_ffmpeg()
    process_audio()