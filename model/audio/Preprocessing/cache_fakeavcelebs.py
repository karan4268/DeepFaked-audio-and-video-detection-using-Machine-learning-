# =============================================================================
# cache_fakeavceleb_audio.py (FIXED — fusion-aligned index)
# =============================================================================
import sys
import os

ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
sys.path.insert(0, ROOT)

import json
import hashlib
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Fusion_Layer.fusion_utility import make_file_id
from extractor import load_audio, config_signature


# =============================================================================
# CONFIG
# =============================================================================

CACHE_ROOT  = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
FAKEAV_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"
FAKEAV_META = os.path.join(FAKEAV_ROOT, "meta_data.csv")

MIN_SAMPLES = 16000
MAX_LEN     = 96000


# =============================================================================
# COMMON
# =============================================================================

def process_audio(audio: np.ndarray) -> np.ndarray | None:
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        return None
    if len(audio) < MIN_SAMPLES:
        return None
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def atomic_save(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


# =============================================================================
# CACHE PATH
# FIX: file field is now a relative path "fakeav_test/{label_dir}/{hid}.npy"
#      so fusion layer can resolve: os.path.join(CACHE_ROOT, entry["file"])
# =============================================================================

def fakeav_cache_path(file_id: str, label: int):
    label_dir = "real" if label == 0 else "fake"
    signature = f"fakeav|{file_id}|{config_signature()}"
    hid       = hashlib.md5(signature.encode()).hexdigest()

    # Relative path from CACHE_ROOT — used directly in the index
    rel_path  = os.path.join("fakeav_test", label_dir, f"{hid}.npy")
    full_path = os.path.join(CACHE_ROOT, rel_path)

    return full_path, rel_path, hid


# =============================================================================
# PROCESS
# =============================================================================

def process_entry(entry):
    try:
        file_path = entry["full_path"]
        label     = entry["label"]
        speaker   = entry["speaker"]
        method    = entry["method"]

        if not os.path.isfile(file_path):
            return {"skip": "missing"}

        if not file_path.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4")):
            return {"skip": "format"}

        file_id = make_file_id(file_path, FAKEAV_ROOT)

        full_path, rel_path, hid = fakeav_cache_path(file_id, label)

        if not os.path.isfile(full_path):
            audio = load_audio(file_path)
            if audio is None:
                return {"skip": "bad_audio"}
            audio = process_audio(audio)
            if audio is None:
                return {"skip": "invalid_audio"}
            atomic_save(full_path, audio)

        # -----------------------------------------------------------------
        # FIX 1: "file" is now a CACHE-ROOT-RELATIVE path with .npy
        #         → fusion loader: np.load(os.path.join(AUDIO_CACHE_ROOT, entry["file"]))
        # FIX 2: "modality" field added for explicit fusion dispatch
        # -----------------------------------------------------------------
        return {
            "file_id":  file_id,           # join key across modalities
            "file":     rel_path,          # e.g. "fakeav_test/real/abc123.npy"
            "label":    label,
            "speaker":  speaker,
            "attack":   method if label == 1 else "bonafide",
            "dataset":  "fakeav",
            "modality": "audio",
        }

    except Exception as e:
        return {"skip": f"error: {e}"}


# =============================================================================
# BUILD
# =============================================================================

def build_fakeavceleb():
    print("\n===== Building FakeAVCeleb Audio Cache =====")

    if not os.path.isfile(FAKEAV_META):
        raise FileNotFoundError(f"Metadata not found: {FAKEAV_META}")

    df = pd.read_csv(FAKEAV_META)
    df.columns = [col.strip() for col in df.columns]

    df["label"] = df["category"].apply(lambda c: 0 if c in ["A", "C"] else 1)

    dir_col = None
    for col in df.columns:
        if col.lower() not in ["source", "target1", "target2", "method",
                                "category", "type", "race", "gender", "path"]:
            dir_col = col
            break

    if dir_col is None:
        raise ValueError("Directory column not found in metadata CSV")

    print(f"[INFO] Directory column: {dir_col}")

    entries = []
    for _, row in df.iterrows():
        filename  = str(row["path"]).strip()
        directory = str(row[dir_col]).strip()

        if directory.startswith("FakeAVCeleb"):
            directory = directory.replace("FakeAVCeleb", "").lstrip("\\/")

        full_path = os.path.join(FAKEAV_ROOT, directory, filename)

        entries.append({
            "full_path": full_path,
            "label":     int(row["label"]),
            "speaker":   row.get("source", "unknown"),
            "method":    row.get("method", "unknown"),
        })

    print(f"[INFO] Total entries: {len(entries)}")

    workers = max(1, cpu_count() - 1)
    print(f"[INFO] Using {workers} workers")

    index = []
    stats = {}

    with Pool(workers) as pool:
        for r in tqdm(pool.imap_unordered(process_entry, entries),
                      total=len(entries), desc="FakeAVCeleb Audio"):
            if isinstance(r, dict) and "skip" in r:
                key = r["skip"].split(":")[0]
                stats[key] = stats.get(key, 0) + 1
            elif r:
                index.append(r)

    save_path = os.path.join(CACHE_ROOT, "fakeav_test", "audio_index.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(index, f, indent=2)

    real = sum(1 for e in index if e["label"] == 0)
    fake = sum(1 for e in index if e["label"] == 1)

    print("\n===== SUMMARY =====")
    print(f"[DONE] Cached: {len(index)}  (real={real}, fake={fake})")
    print(f"[SKIPPED] {stats}")
    print(f"[INDEX] {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    build_fakeavceleb()
    print("\n✅ FakeAVCeleb audio cache complete.")