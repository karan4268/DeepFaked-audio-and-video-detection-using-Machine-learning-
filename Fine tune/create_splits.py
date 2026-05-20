# =============================================================================
# create_splits.py
#
# Creates a reproducible 200/150/150 speaker split for FakeAVCeleb.
# Run ONCE — all other scripts load from splits.json.
#
# Output:
#   splits.json  {
#     "train_speakers": [...200 ids...],
#     "val_speakers":   [...150 ids...],
#     "test_speakers":  [...150 ids...],
#     "train_indices":  [...],   # index.json positions
#     "val_indices":    [...],
#     "test_indices":   [...],
#   }
# =============================================================================

import os
import json
import numpy as np
from collections import defaultdict

# =============================================================================
INDEX_JSON  = r"D:\FakeAVCache\Video\video_index.json"# both index files have same order, so just load one
OUTPUT_PATH = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\splits.json"
)
SEED = 42
# =============================================================================

def main():
    np.random.seed(SEED)

    print("[INFO] Loading index.json ...")
    with open(INDEX_JSON) as f:
        index = json.load(f)

    # Get all unique speakers
    speakers = sorted(set(s["speaker"] for s in index))
    print(f"  Total speakers : {len(speakers)}")
    print(f"  Total samples  : {len(index)}")

    # Shuffle speakers
    speakers = np.array(speakers)
    np.random.shuffle(speakers)

    # Split speakers 200 / 150 / 150
    train_speakers = set(speakers[:200].tolist())
    val_speakers   = set(speakers[200:350].tolist())
    test_speakers  = set(speakers[350:].tolist())

    assert len(train_speakers) == 200
    assert len(val_speakers)   == 150
    assert len(test_speakers)  == 150
    assert len(train_speakers & val_speakers & test_speakers) == 0  # no overlap

    # Assign each sample to a split based on its speaker
    train_indices, val_indices, test_indices = [], [], []

    for i, s in enumerate(index):
        spk = s["speaker"]
        if spk in train_speakers:
            train_indices.append(i)
        elif spk in val_speakers:
            val_indices.append(i)
        elif spk in test_speakers:
            test_indices.append(i)

    # Stats
    def split_stats(indices, name):
        samples  = [index[i] for i in indices]
        n_real   = sum(1 for s in samples if s["label"] == 0)
        n_fake   = sum(1 for s in samples if s["label"] == 1)
        speakers = len(set(s["speaker"] for s in samples))
        print(f"  {name:<6} : {len(samples):5d} samples  "
              f"({n_real} real, {n_fake} fake)  {speakers} speakers")

    print("\n[INFO] Split summary:")
    split_stats(train_indices, "Train")
    split_stats(val_indices,   "Val")
    split_stats(test_indices,  "Test")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    splits = {
        "seed":            SEED,
        "train_speakers":  sorted(train_speakers),
        "val_speakers":    sorted(val_speakers),
        "test_speakers":   sorted(test_speakers),
        "train_indices":   train_indices,
        "val_indices":     val_indices,
        "test_indices":    test_indices,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n[INFO] Saved splits → {OUTPUT_PATH}")
    print("[INFO] All other scripts will load from this file.")


if __name__ == "__main__":
    main()
