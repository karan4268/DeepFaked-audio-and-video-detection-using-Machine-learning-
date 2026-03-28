import os
import json
import random
import hashlib

random.seed(42)  # For reproducibility

ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"
CACHE_DIR = os.path.join(ROOT_DIR, "cached_faces")

TARGET_PER_CLASS = {
    "RealVideo-RealAudio": 500,
    "RealVideo-FakeAudio": 500,
    "FakeVideo-RealAudio": 1000,
    "FakeVideo-FakeAudio": 1000,
}

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20


def video_cache_dir(video_path):
    rel = os.path.relpath(video_path, ROOT_DIR)
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, h)


def collect_identities(class_path):
    id_videos = {}
    for root, dirs, files in os.walk(class_path):
        if os.path.basename(root).startswith("id"):
            vids = [os.path.join(root, f) for f in files if f.endswith(".mp4")]
            if vids:
                id_videos[root] = vids
    return id_videos


def split_ids_by_identity(ids, target_count):
    """
    Shuffle IDs and select a subset up to target_count videos,
    distributing them proportionally across train/val/test.
    """
    random.shuffle(ids)

    # Split fractions
    n_train = int(len(ids) * TRAIN_FRAC)
    n_val = int(len(ids) * VAL_FRAC)
    n_test = len(ids) - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def build_dataset():
    split_json = {"train_full_pool": [], "val": [], "test": []}

    for class_name, total_target in TARGET_PER_CLASS.items():
        print(f"\nProcessing {class_name}")

        class_path = os.path.join(ROOT_DIR, class_name)
        id_videos = collect_identities(class_path)
        ids = list(id_videos.keys())
        if not ids:
            print(f"No IDs found for {class_name}, skipping")
            continue

        # Split IDs by identity to avoid leakage
        train_ids, val_ids, test_ids = split_ids_by_identity(ids, total_target)

        split_targets = {
            "train_full_pool": int(total_target * TRAIN_FRAC),
            "val": int(total_target * VAL_FRAC),
            "test": total_target - int(total_target * TRAIN_FRAC) - int(total_target * VAL_FRAC)
        }

        for split_name, id_list in zip(["train_full_pool", "val", "test"],
                                       [train_ids, val_ids, test_ids]):
            count = 0
            target = split_targets[split_name]

            for id_dir in id_list:
                for vid in id_videos[id_dir]:
                    if count >= target:
                        break

                    cache_path = video_cache_dir(vid)
                    if not os.path.isdir(cache_path):
                        continue

                    label_file = os.path.join(cache_path, "label.txt")
                    if not os.path.exists(label_file):
                        continue

                    with open(label_file) as f:
                        primary, binary = map(int, f.read().strip().split(","))

                    split_json[split_name].append({
                        "path": cache_path,
                        "primary_label": primary,
                        "binary_label": binary
                    })
                    count += 1

                if count >= target:
                    break

            print(f"{split_name}: selected {count} videos")

    out_path = os.path.join(ROOT_DIR, "dataset_split_new.json")
    with open(out_path, "w") as f:
        json.dump(split_json, f, indent=2)

    print("\nDataset split saved:", out_path)


if __name__ == "__main__":
    build_dataset()