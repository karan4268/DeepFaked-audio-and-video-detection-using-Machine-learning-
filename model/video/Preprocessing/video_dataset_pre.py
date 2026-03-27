# video_dataset_pre_optimized.py
# ------------------------------------------------------------
# FaceForensics++ C23 Identity-Level Split (Optimized)
#
# Fixes:
# ✔ Eliminates O(N²) fake scanning (uses indexing)
# ✔ Adds dataset validation
# ✔ Supports hardlink/symlink/copy
# ✔ Handles missing files safely
# ✔ Faster + scalable for full FF++
# ------------------------------------------------------------

import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# =========================
# Paths
# =========================
source_root = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\ff++c23\FaceForensics++_C23"
dest_root   = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video"

# Dataset folders
real_folder = "original"
fake_folders = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

# Split sizes (IDENTITIES)
train_ids = 600
val_ids   = 200
test_ids  = 200

# File handling: "copy", "hardlink", "symlink"
FILE_MODE = "hardlink" # "copy" is safest but uses more space; "hardlink" is efficient on same filesystem; "symlink" can break if moved

random.seed(42)

# =========================
# Helper: file operation
# =========================
def transfer_file(src, dst):
    if os.path.exists(dst):
        return
    try:
        if FILE_MODE == "copy":
            shutil.copy2(src, dst)
        elif FILE_MODE == "hardlink":
            os.link(src, dst)
        elif FILE_MODE == "symlink":
            os.symlink(src, dst)
        else:
            raise ValueError("Invalid FILE_MODE")
    except Exception as e:
        print(f"[ERROR] {src} -> {dst} | {e}")

# =========================
# Create destination folders
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dest_root, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, split, "fake"), exist_ok=True)

# =========================
# Step 1: Collect IDs
# =========================
print("[INFO] Collecting identities...")

original_path = os.path.join(source_root, real_folder)

all_ids = [
    os.path.splitext(f)[0]
    for f in os.listdir(original_path)
    if f.endswith(".mp4")
]

total_required = train_ids + val_ids + test_ids

assert len(all_ids) >= total_required, \
    f"Not enough identities: required {total_required}, found {len(all_ids)}"

random.shuffle(all_ids)

selected_ids = all_ids[:total_required]

splits = {
    "train": selected_ids[:train_ids],
    "val":   selected_ids[train_ids:train_ids + val_ids],
    "test":  selected_ids[train_ids + val_ids:]
}

print(f"[INFO] Selected identities: {len(selected_ids)}")

# =========================
# Step 2: Build fake index (CRITICAL FIX)
# =========================
print("[INFO] Indexing fake videos...")

fake_index = defaultdict(list)

for fake_type in fake_folders:
    folder_path = os.path.join(source_root, fake_type)

    if not os.path.exists(folder_path):
        print(f"[WARN] Missing folder: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            vid_id = file.split("_")[0]   # extract identity
            fake_index[vid_id].append(os.path.join(folder_path, file))

print(f"[INFO] Indexed identities with fakes: {len(fake_index)}")

# =========================
# Step 3: Copy/Link files
# =========================
stats = {
    "train": {"real": 0, "fake": 0},
    "val":   {"real": 0, "fake": 0},
    "test":  {"real": 0, "fake": 0},
}

print("[INFO] Processing dataset...")

for split_name, ids in splits.items():

    print(f"\n[INFO] {split_name.upper()} split")

    for vid_id in tqdm(ids):

        # ---------- REAL ----------
        real_src = os.path.join(source_root, real_folder, vid_id + ".mp4")

        if os.path.exists(real_src):
            dst = os.path.join(dest_root, split_name, "real", vid_id + ".mp4")
            transfer_file(real_src, dst)
            stats[split_name]["real"] += 1
        else:
            print(f"[WARN] Missing real: {vid_id}")

        # ---------- FAKES ----------
        fake_list = fake_index.get(vid_id, [])

        if not fake_list:
            print(f"[WARN] No fake for ID: {vid_id}")

        for fake_src in fake_list:
            fname = os.path.basename(fake_src)
            dst = os.path.join(dest_root, split_name, "fake", fname)

            transfer_file(fake_src, dst)
            stats[split_name]["fake"] += 1

# =========================
# Stats
# =========================
print("\n================ DATASET STATS ================\n")

for split in ["train", "val", "test"]:
    real_count = stats[split]["real"]
    fake_count = stats[split]["fake"]

    ratio = fake_count / real_count if real_count else 0

    print(f"{split.upper()} SET:")
    print(f"  Real videos : {real_count}")
    print(f"  Fake videos : {fake_count}")
    print(f"  Fake/Real   : {ratio:.2f}")
    print("-" * 40)

print("\n[INFO] Done.")

#===========================================================output=========================================================================

#[INFO] Collecting identities...
#[INFO] Selected identities: 1000
#[INFO] Indexing fake videos...
#[INFO] Indexed identities with fakes: 1028
#[INFO] Processing dataset...

#[INFO] TRAIN split
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:00<00:00, 1281.64it/s]

#[INFO] VAL split
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 1292.70it/s]

#[INFO] TEST split
#100%|████████████████████████████████████████ ██████| 200/200 [00:00<00:00, 1284.61it/s]

#================ DATASET STATS ================

#TRAIN SET:
#  Real videos : 600
#  Fake videos : 3000
#  Fake/Real   : 5.00
#----------------------------------------
#VAL SET:
#  Real videos : 200
#  Fake videos : 1000
#  Fake/Real   : 5.00
#----------------------------------------
#TEST SET:
#  Real videos : 200
#  Fake videos : 1000
#  Fake/Real   : 5.00
#----------------------------------------

#[INFO] Done.