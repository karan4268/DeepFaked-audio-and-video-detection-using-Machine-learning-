# cache_audio.py
# =============================================================================
# ASVspoof2019 LA - Identity-Aware Waveform Cache (INT16)
# =============================================================================
#
# NEW FEATURES:
#   ✅ Stores speaker ID
#   ✅ Stores attack ID (A01–A19 / bonafide)
#   ✅ Stores original file_id
#   ✅ Enables attack-wise and speaker-wise evaluation later
#
# NOTE:
#   You DO NOT need to rebuild audio if already cached.
#   Only index.json becomes richer.
# =============================================================================

import os
import json
import hashlib
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from Preprocessing.extractor import load_audio, config_signature


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\AVSpoof 2019\LA\LA"

PROTOCOL_ROOT = os.path.join(ROOT_DIR, "ASVspoof2019_LA_cm_protocols")

CACHE_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"

SPLITS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "val":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "test":  "ASVspoof2019.LA.cm.eval.trl.txt"
}

AUDIO_FOLDERS = {
    "train": "ASVspoof2019_LA_train",
    "val":   "ASVspoof2019_LA_dev",
    "test":  "ASVspoof2019_LA_eval"
}

AUDIO_EXT = ".flac"


# ------------------------------------------------------------
# PROTOCOL LOADER (IDENTITY-AWARE)
# ------------------------------------------------------------

def load_protocol(protocol_path):
    data = []

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            # ASVspoof2019 format
            speaker_id = parts[0]
            file_id    = parts[1]
            attack_id  = parts[3] if parts[-1] != "bonafide" else "bonafide"
            label      = 0 if parts[-1] == "bonafide" else 1

            data.append({
                "speaker": speaker_id,
                "file_id": file_id,
                "attack": attack_id,
                "label": label
            })

    return data


# ------------------------------------------------------------
# CACHE PATH (UNCHANGED)
# ------------------------------------------------------------

def cache_path(file_id, split, label):
    label_dir = "real" if label == 0 else "fake"

    signature = f"{file_id}|{split}|{config_signature()}"
    hash_id   = hashlib.md5(signature.encode()).hexdigest()

    return os.path.join(
        CACHE_ROOT, split, label_dir, f"{hash_id}.npy"
    ), hash_id


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def process_file(entry):
    try:
        file_id    = entry["file_id"]
        label      = entry["label"]
        split      = entry["split"]
        speaker_id = entry["speaker"]
        attack_id  = entry["attack"]

    except Exception as e:
        print(f"[ARG ERROR] {entry} | {e}")
        return None

    audio_path = os.path.join(
        ROOT_DIR,
        AUDIO_FOLDERS[split],
        "flac",
        f"{file_id}{AUDIO_EXT}"
    )

    path, hid = cache_path(file_id, split, label)

    if not os.path.isfile(audio_path):
        return None

    try:
        # Only process if not already cached
        if not os.path.isfile(path):
            audio = load_audio(audio_path)

            # INT16 conversion
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            tmp = path + ".tmp.npy"
            np.save(tmp, audio)
            os.replace(tmp, path)

        # ✅ RETURN FULL METADATA
        return {
            "file": f"{hid}.npy",
            "label": label,
            "speaker": speaker_id,
            "attack": attack_id,
            "file_id": file_id
        }

    except Exception as e:
        print(f"[ERROR] {file_id}: {e}")
        return None


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Building ID-aware waveform cache...")

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} workers")

    for split, proto in SPLITS.items():
        print(f"\nSplit: {split}")

        protocol_path = os.path.join(PROTOCOL_ROOT, proto)
        data          = load_protocol(protocol_path)

        print(f"Total files: {len(data)}")

        # Attach split info
        for d in data:
            d["split"] = split

        index = []

        with Pool(workers) as pool:
            for r in tqdm(pool.imap_unordered(process_file, data), total=len(data)):
                if r is not None:
                    index.append(r)

        print(f"[DEBUG] Valid cached: {len(index)} / {len(data)}")

        index_path = os.path.join(CACHE_ROOT, split, "index.json")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[DONE] {split}: {len(index)} samples")

    print("\n✅ Identity-aware cache/index ready.")


# ------------------------------------------------------------
# WEIGHTED SAMPLER (UNCHANGED)
# ------------------------------------------------------------

def make_audio_sampler(index: list):
    from torch.utils.data import WeightedRandomSampler
    import torch
    import numpy as np

    labels        = np.array([entry["label"] for entry in index])
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )