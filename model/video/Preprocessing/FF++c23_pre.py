# video_dataset_pre_optimized.py
# ------------------------------------------------------------
# FaceForensics++ C23 Identity-Level Split (Optimized)
#
# CHANGES:
#   - Uses ALL available identities instead of capping at 1000
#   - 70/15/15 split (train/val/test) instead of fixed 600/200/200
#   - Same structure as CelebDF: train/real, train/fake, val/real etc.
#   - Cache-friendly: DeepfakedDataset + face_cache.py will work as-is
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
dest_root   = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23"

real_folder  = "original"
fake_folders = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

# 70 / 15 / 15 split across ALL available identities
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15 

# "copy", "hardlink", "symlink"
FILE_MODE = "hardlink"

random.seed(42)

# =========================
# Helper
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
            raise ValueError(f"Invalid FILE_MODE: {FILE_MODE}")
    except Exception as e:
        print(f"[ERROR] {src} -> {dst} | {e}")

# =========================
# Create destination folders
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dest_root, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, split, "fake"), exist_ok=True)

# =========================
# Step 1: Collect ALL identities
# =========================
print("[INFO] Collecting identities...")

original_path = os.path.join(source_root, real_folder)

all_ids = [
    os.path.splitext(f)[0]
    for f in os.listdir(original_path)
    if f.endswith(".mp4")
]

print(f"[INFO] Total identities found: {len(all_ids)}")

random.shuffle(all_ids)

# Compute split sizes from ratios
n_total = len(all_ids)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val   # remainder goes to test

splits = {
    "train": all_ids[:n_train],
    "val":   all_ids[n_train : n_train + n_val],
    "test":  all_ids[n_train + n_val:]
}

print(f"[INFO] Split sizes — "
      f"train: {n_train}  val: {n_val}  test: {n_test}  "
      f"(total: {n_train + n_val + n_test})")

# =========================
# Step 2: Build fake index
# =========================
print("\n[INFO] Indexing fake videos...")

fake_index = defaultdict(list)

for fake_type in fake_folders:
    folder_path = os.path.join(source_root, fake_type)
    if not os.path.exists(folder_path):
        print(f"[WARN] Missing folder: {folder_path}")
        continue
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            vid_id = file.split("_")[0]
            fake_index[vid_id].append(os.path.join(folder_path, file))

print(f"[INFO] Identities with fakes indexed: {len(fake_index)}")

# Warn if any identity has fewer than 6 fakes (one per method)
missing_methods = [
    vid_id for vid_id in all_ids
    if len(fake_index.get(vid_id, [])) < len(fake_folders)
]
if missing_methods:
    print(f"[WARN] {len(missing_methods)} identities have fewer than "
          f"{len(fake_folders)} fake methods")

# =========================
# Step 3: Transfer files
# =========================
stats = {s: {"real": 0, "fake": 0} for s in ["train", "val", "test"]}

print("\n[INFO] Processing splits...")

for split_name, ids in splits.items():
    print(f"\n[INFO] {split_name.upper()} — {len(ids)} identities")

    for vid_id in tqdm(ids):

        # REAL
        real_src = os.path.join(source_root, real_folder, vid_id + ".mp4")
        if os.path.exists(real_src):
            dst = os.path.join(dest_root, split_name, "real", vid_id + ".mp4")
            transfer_file(real_src, dst)
            stats[split_name]["real"] += 1
        else:
            print(f"[WARN] Missing real: {vid_id}")

        # FAKES
        fake_list = fake_index.get(vid_id, [])
        if not fake_list:
            print(f"[WARN] No fakes for ID: {vid_id}")
        for fake_src in fake_list:
            # Prefix with method name to avoid filename collisions —
            # all 5 standard methods share identical filenames (000_003.mp4 etc)
            # e.g. Deepfakes__000_003.mp4, Face2Face__000_003.mp4
            method = os.path.basename(os.path.dirname(fake_src))
            fname  = f"{method}__{os.path.basename(fake_src)}"
            dst    = os.path.join(dest_root, split_name, "fake", fname)
            transfer_file(fake_src, dst)
            stats[split_name]["fake"] += 1

# =========================
# Stats
# =========================
print("\n================ DATASET STATS ================\n")

total_real = total_fake = 0
for split in ["train", "val", "test"]:
    r = stats[split]["real"]
    f = stats[split]["fake"]
    total_real += r
    total_fake += f
    print(f"{split.upper():5s} — real: {r:>5}  fake: {f:>5}  "
          f"ratio: {f/r:.2f}  total: {r+f}")

print("-" * 45)
print(f"TOTAL — real: {total_real:>5}  fake: {total_fake:>5}  "
      f"ratio: {total_fake/total_real:.2f}  total: {total_real+total_fake}")
print("\n[INFO] Done.")
print("\nNext step: run face_cache_ffpp.py")
print(f"  ROOT_DIR is already set to: {dest_root}")


# [INFO] Collecting identities...
#[INFO] Total identities found: 1000
#[INFO] Split sizes — train: 700  val: 150  test: 150  (total: 1000)

#[INFO] Indexing fake videos...
#[INFO] Identities with fakes indexed: 1028
#[WARN] 1000 identities have fewer than 6 fake methods

#[INFO] Processing splits...

#[INFO] TRAIN — 700 identities
#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| #700/700 [00:00<00:00, 968.75it/s]

#[INFO] VAL — 150 identities
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:00<00:00, 1123.04it/s]

#[INFO] TEST — 150 identities
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:00<00:00, 1457.23it/s]

#================ DATASET STATS ================

#TRAIN — real:   700  fake:  3500  ratio: 5.00  total: 4200
#VAL   — real:   150  fake:   750  ratio: 5.00  total: 900
#TEST  — real:   150  fake:   750  ratio: 5.00  total: 900
#---------------------------------------------
#TOTAL — real:  1000  fake:  5000  ratio: 5.00  total: 6000

#[INFO] Done.

#Next step: run face_cache.py with ROOT_DIR pointing at:
#  F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23