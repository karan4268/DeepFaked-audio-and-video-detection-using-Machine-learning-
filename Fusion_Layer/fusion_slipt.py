# =============================================================================
# build_fusion_index.py (FIXED — CLEAN & DETERMINISTIC)
# =============================================================================

import os
import json
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

INPUT_INDEX  = r"D:\FakeAVCache\Video\index.json"
OUTPUT_INDEX = r"D:\FakeAVCache\Video\fusion_index.json"

# =============================================================================


def infer_labels(meta):
    """
    Strict modality inference using ONLY dataset structure.
    This is the ground truth in FakeAVCeleb.
    """

    path = meta.get("file_id", "").lower()

    if not path:
        raise ValueError(f"Missing file_id in meta: {meta}")

    # Extract top-level folder
    top = path.split("/")[0]

    if top == "realvideo-realaudio":
        return 0, 0
    elif top == "fakevideo-realaudio":
        return 1, 0
    elif top == "realvideo-fakeaudio":
        return 0, 1
    elif top == "fakevideo-fakeaudio":
        return 1, 1
    else:
        raise ValueError(f"Unknown top-level folder: {top}")


def main():

    print("[INFO] Loading index ...")
    with open(INPUT_INDEX) as f:
        index = json.load(f)

    fusion_index = []

    stats = {
        "real_real": 0,
        "fake_real": 0,
        "real_fake": 0,
        "fake_fake": 0
    }

    print("[INFO] Building fusion index ...")

    for i, meta in enumerate(tqdm(index)):

        v_label, a_label = infer_labels(meta)
        fusion_label     = int(v_label or a_label)

        # Copy original meta
        new_meta = dict(meta)

        new_meta["index"] = i

        # Add fusion fields
        new_meta["video_label"]  = v_label
        new_meta["audio_label"]  = a_label
        new_meta["fusion_label"] = fusion_label

        fusion_index.append(new_meta)

        # Stats
        if v_label == 0 and a_label == 0:
            stats["real_real"] += 1
        elif v_label == 1 and a_label == 0:
            stats["fake_real"] += 1
        elif v_label == 0 and a_label == 1:
            stats["real_fake"] += 1
        elif v_label == 1 and a_label == 1:
            stats["fake_fake"] += 1

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    with open(OUTPUT_INDEX, "w") as f:
        json.dump(fusion_index, f, indent=2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(" Fusion Index Summary (FIXED)")
    print("="*50)

    total = sum(stats.values())

    for k, v in stats.items():
        print(f"  {k:<12}: {v:6d} ({v/total:.2%})")

    print("-"*50)
    print(f"  Total       : {total}")

    print("\n[INFO] Saved →", OUTPUT_INDEX)


if __name__ == "__main__":
    main()