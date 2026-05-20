# cache_audio.py  (Preprocessing/cache_audio.py)
# =============================================================================
# Unified Audio Cache Builder (ASVspoof 2019 LA + In-the-Wild)
#
# FIXES vs previous version:
#   [FIX 1] ITW MAX_LEN raised from 64000 (4s) to 96000 (6s).
#           ITW clips average 8–15s and voice-conversion/TTS artifacts
#           often concentrate in the middle of an utterance, not just the
#           first 4s. 6s is a practical upper bound that fits comfortably
#           in GPU memory at batch_size=16 after mel extraction.
#           Change MAX_LEN at the top of this file to adjust globally.
#
#   [FIX 2] ITW build_itw() now uses multiprocessing (Pool) matching the
#           ASVspoof pipeline. The original single-process loop over ~30k
#           files took 3–4× longer than necessary.
#
#   [FIX 3] meta.csv label column mapped defensively: any value that is not
#           "bona-fide" is treated as spoof (label=1). Previously an
#           unexpected value would silently produce NaN → int crash.
#
# NOTE: Re-running this script on an already-cached dataset is safe.
#       Existing .npy files are skipped (checked before load_audio call).
#       Only index.json is always rewritten.
# =============================================================================

import os
import json
import hashlib
import random
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from extractor import load_audio, config_signature


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

CACHE_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"

MIN_SAMPLES = 16000        # 1 second minimum — shorter clips are noise
MAX_LEN     = 96000        # [FIX 1] 6 seconds (was 4s / 64000)


# =============================================================================
# ASVSPOOF CONFIG
# =============================================================================

ASV_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\AVSpoof 2019\LA\LA"

ASV_PROTOCOL_ROOT = os.path.join(ASV_ROOT, "ASVspoof2019_LA_cm_protocols")

ASV_SPLITS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "val":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "test":  "ASVspoof2019.LA.cm.eval.trl.txt"
}

ASV_AUDIO_FOLDERS = {
    "train": "ASVspoof2019_LA_train",
    "val":   "ASVspoof2019_LA_dev",
    "test":  "ASVspoof2019_LA_eval"
}


# =============================================================================
# ITW CONFIG
# =============================================================================

ITW_ROOT   = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\in_the_wild\release_in_the_wild"

VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SEED       = 42


# =============================================================================
# COMMON AUDIO PROCESSING
# =============================================================================

def process_audio(audio: np.ndarray) -> np.ndarray | None:
    """Validate, trim, and convert to int16 for compact storage."""
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        return None
    if len(audio) < MIN_SAMPLES:
        return None

    # [FIX 1] Trim to MAX_LEN (6s by default)
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def atomic_save(path: str, array: np.ndarray) -> None:
    """Write via tmp file then rename to avoid partial writes on crash."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


# =============================================================================
# ASVSPOOF
# =============================================================================

def load_asv_protocol(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker = parts[0]
            file_id = parts[1]
            attack  = parts[3] if parts[-1] != "bonafide" else "bonafide"
            label   = 0 if parts[-1] == "bonafide" else 1
            data.append({
                "speaker": speaker,
                "file_id": file_id,
                "attack":  attack,
                "label":   label
            })
    return data


def asv_cache_path(file_id: str, split: str, label: int) -> tuple[str, str]:
    label_dir = "real" if label == 0 else "fake"
    signature = f"asvspoof|{split}|{file_id}|{config_signature()}"
    hid       = hashlib.md5(signature.encode()).hexdigest()
    path      = os.path.join(CACHE_ROOT, split, label_dir, f"{hid}.npy")
    return path, hid


def process_asv(entry: dict) -> dict | None:
    try:
        file_id = entry["file_id"]
        label   = entry["label"]
        split   = entry["split"]

        audio_path = os.path.join(
            ASV_ROOT, ASV_AUDIO_FOLDERS[split], "flac", f"{file_id}.flac"
        )

        if not os.path.isfile(audio_path):
            return None

        path, hid = asv_cache_path(file_id, split, label)

        if not os.path.isfile(path):
            audio = load_audio(audio_path)
            audio = process_audio(audio)
            if audio is None:
                return None
            atomic_save(path, audio)

        return {
            "file":    f"{hid}.npy",
            "label":   label,
            "speaker": entry["speaker"],
            "attack":  entry["attack"],
            "dataset": "asvspoof",
            "file_id": file_id
        }

    except Exception as e:
        print(f"[ASV ERROR] {entry.get('file_id', '?')}: {e}")
        return None


def build_asvspoof() -> None:
    print("\n===== Building ASVspoof Cache =====")
    workers = max(1, cpu_count() - 1)
    print(f"[ASV] Using {workers} workers")

    for split, proto in ASV_SPLITS.items():
        print(f"\n[ASV] Split: {split}")

        data = load_asv_protocol(os.path.join(ASV_PROTOCOL_ROOT, proto))
        for d in data:
            d["split"] = split

        index = []
        with Pool(workers) as pool:
            for r in tqdm(
                pool.imap_unordered(process_asv, data),
                total=len(data),
                desc=f"ASV {split}"
            ):
                if r:
                    index.append(r)

        save_path = os.path.join(CACHE_ROOT, split, "index.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(index, f, indent=2)

        real = sum(1 for e in index if e["label"] == 0)
        fake = sum(1 for e in index if e["label"] == 1)
        print(f"[ASV] {split}: {len(index)} cached  (real={real}, fake={fake})")


# =============================================================================
# ITW — multiprocessing 
# =============================================================================

def split_speakers(speakers: list) -> tuple[set, set, set]:
    random.seed(SEED)
    speakers = sorted(speakers)
    random.shuffle(speakers)

    n      = len(speakers)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)

    test  = set(speakers[:n_test])
    val   = set(speakers[n_test:n_test + n_val])
    train = set(speakers[n_test + n_val:])
    return train, val, test


def itw_cache_path(file_id: str, split: str, label: int) -> tuple[str, str]:
    label_dir = "real" if label == 0 else "fake"
    signature = f"itw|{split}|{file_id}|{config_signature()}"
    hid       = hashlib.md5(signature.encode()).hexdigest()
    path      = os.path.join(CACHE_ROOT, f"itw_{split}", label_dir, f"{hid}.npy")
    return path, hid


def process_itw(entry: dict) -> dict | None:
    """Worker function for ITW multiprocessing. [FIX 2]"""
    try:
        file_id  = entry["file_id"]
        label    = entry["label"]
        split    = entry["split"]
        speaker  = entry["speaker"]

        audio_path = os.path.join(ITW_ROOT, file_id)
        if not os.path.isfile(audio_path):
            return None

        path, hid = itw_cache_path(file_id, split, label)

        if not os.path.isfile(path):
            audio = load_audio(audio_path)
            audio = process_audio(audio)
            if audio is None:
                return None
            atomic_save(path, audio)

        return {
            "file":    f"{hid}.npy",
            "label":   label,
            "speaker": speaker,
            "attack":  f"itw_{speaker}" if label == 1 else "bonafide",
            "dataset": "itw",
            "file_id": file_id
        }

    except Exception as e:
        print(f"[ITW ERROR] {entry.get('file_id', '?')}: {e}")
        return None


def build_itw() -> None:
    print("\n===== Building ITW Cache =====")

    meta_path = os.path.join(ITW_ROOT, "meta.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.csv not found at {meta_path}")

    df = pd.read_csv(meta_path)

    # [FIX 3] Defensive label mapping — anything not "bona-fide" → spoof
    df["label"] = df["label"].apply(
        lambda v: 0 if str(v).strip().lower() == "bona-fide" else 1
    )

    train_spk, val_spk, test_spk = split_speakers(df["speaker"].unique().tolist())

    def assign_split(s):
        if s in test_spk: return "test"
        if s in val_spk:  return "val"
        return "train"

    df["split"] = df["speaker"].apply(assign_split)

    workers = max(1, cpu_count() - 1)
    print(f"[ITW] Using {workers} workers")

    for split in ["train", "val", "test"]:
        print(f"\n[ITW] Split: {split}")

        sub = df[df["split"] == split]

        entries = [
            {
                "file_id": row["file"],
                "label":   int(row["label"]),
                "speaker": row["speaker"],
                "split":   split
            }
            for _, row in sub.iterrows()
        ]

        index = []

        # [FIX 2] Use Pool instead of single-process loop
        with Pool(workers) as pool:
            for r in tqdm(
                pool.imap_unordered(process_itw, entries),
                total=len(entries),
                desc=f"ITW {split}"
            ):
                if r:
                    index.append(r)

        save_path = os.path.join(CACHE_ROOT, f"itw_{split}", "index.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(index, f, indent=2)

        real = sum(1 for e in index if e["label"] == 0)
        fake = sum(1 for e in index if e["label"] == 1)
        print(f"[ITW] {split}: {len(index)} cached  (real={real}, fake={fake})")


# =============================================================================
# WEIGHTED SAMPLER FACTORY  (unchanged, re-exported for train_audio.py)
# =============================================================================

def make_audio_sampler(index: list):
    from torch.utils.data import WeightedRandomSampler
    import torch

    labels        = np.array([entry["label"] for entry in index])
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    build_asvspoof()
    build_itw()
    print("\n✅ cache for (ASVspoof+ITW) complete.")