# =============================================================================
# create_balanced_fakeav_video_subset.py (ADAPTED FOR YOUR JSON)
# =============================================================================

import json
import random
from collections import Counter

# ---------------- CONFIG ----------------
INPUT_JSON = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2\dataset_split_fixed.json"
OUTPUT_JSON = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\fakeav_balanced_rr_fr.json"
SAMPLES_PER_CLASS = 500
SEED = 42

random.seed(SEED)

RR_LABEL = 0  # RealVideo-RealAudio
FR_LABEL = 1  # FakeVideo-RealAudio


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():

    data = load_json(INPUT_JSON)

    # 🔥 Merge all splits
    all_samples = data["train"] + data["val"] + data["test"]

    print(f"[INFO] Total samples (all splits): {len(all_samples)}")

    rr_samples = []
    fr_samples = []

    # 🔥 Filter only RR and FR
    for sample in all_samples:
        label = sample["primary_label"]

        if label == RR_LABEL:
            rr_samples.append(sample)

        elif label == FR_LABEL:
            fr_samples.append(sample)

    print(f"[INFO] Total RR: {len(rr_samples)}")
    print(f"[INFO] Total FR: {len(fr_samples)}")

    # Shuffle
    random.shuffle(rr_samples)
    random.shuffle(fr_samples)

    # Balance
    n = min(SAMPLES_PER_CLASS, len(rr_samples), len(fr_samples))

    rr_selected = rr_samples[:n]
    fr_selected = fr_samples[:n]

    balanced_samples = rr_selected + fr_selected
    random.shuffle(balanced_samples)

    print(f"[INFO] Final balanced dataset: {len(balanced_samples)} ({n} per class)")

    # Debug distribution
    labels = [s["primary_label"] for s in balanced_samples]
    print("[INFO] Final distribution:", Counter(labels))

    # Save
    output_data = {
        "test": balanced_samples   # keep "test" key for compatibility
    }

    save_json(output_data, OUTPUT_JSON)

    print(f"✅ Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()