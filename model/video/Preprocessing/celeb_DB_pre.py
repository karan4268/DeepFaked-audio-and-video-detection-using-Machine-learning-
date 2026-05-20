# celebdf_prepare.py
# =============================================================================
# CelebDF-v2 Dataset Preparation
#
# Reads videos from original CelebDF folder structure and creates a clean
# train/val/test split matching the existing FF++ pipeline format:
#
#   Output root/
#       train/real/  train/fake/
#       val/real/    val/fake/
#       test/real/   test/fake/
#
# SPLIT STRATEGY:
#   - Test  → exactly the videos listed in List_of_testing_videos.txt
#   - Train/Val → remaining videos split by SUBJECT ID (not random video split)
#                 80% of subjects → train, 20% → val
#                 Prevents same face appearing in both splits
#
# LABEL CONVENTION (CelebDF txt uses inverted labels):
#   txt file:  1 = real,  0 = fake
#   pipeline:  0 = real,  1 = fake   ← we convert to this
#
# FILE MODE:
#   hardlink → no disk space wasted, instant, works on same filesystem
#   If hardlink fails (cross-drive), falls back to symlink then copy
# =============================================================================

from modulefinder import test
import os
import re
import random
import shutil
from collections import defaultdict

# =============================================================================
# CONFIG — edit these paths
# =============================================================================

CELEB_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\celeb DB"

OUTPUT_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF"

TEST_LIST_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\celeb DB\List_of_testing_videos.txt"

# Source folder names inside CELEB_ROOT
REAL_FOLDERS = ["Celeb-real", "YouTube-real"]
FAKE_FOLDERS = ["Celeb-synthesis"]

# Split ratio for remaining (non-test) videos
TRAIN_RATIO = 0.80

# Random seed for reproducibility
SEED = 42

# =============================================================================
# HELPERS
# =============================================================================

def transfer_file(src, dst):
    """Try hardlink → symlink → copy. Never overwrites existing files."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def extract_subject_id(filename):
    """
    Extract primary subject ID from filename.

    Real:    id1_0007.mp4       → 'id1'
             00170.mp4          → 'yt_00170'  (YouTube-real, no subject)
    Fake:    id1_id0_0007.mp4   → 'id1'  (target identity, i.e. whose face is shown)
    """
    base = os.path.splitext(filename)[0]

    # Celeb-synthesis: id1_id0_0007 → target is id1
    m = re.match(r'^(id\d+)_id\d+_\d+$', base)
    if m:
        return m.group(1)

    # Celeb-real: id1_0007 → id1
    m = re.match(r'^(id\d+)_\d+$', base)
    if m:
        return m.group(1)

    # YouTube-real: 00170 → unique per-video subject
    return f"yt_{base}"


def load_test_list(txt_path):
    """
    Returns set of (relative_path, pipeline_label) from the txt file.
    Converts CelebDF convention (1=real, 0=fake) to pipeline (0=real, 1=fake).

    txt format:  '1 YouTube-real/00170.mp4'
    """
    test_set = {}   # relative_path → pipeline_label

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            celeb_label = int(parts[0])   # 1=real, 0=fake in CelebDF
            rel_path    = parts[1].strip()

            # Convert to pipeline label
            pipeline_label = 0 if celeb_label == 1 else 1

            test_set[rel_path] = pipeline_label

    return test_set


def collect_all_videos(celeb_root, real_folders, fake_folders):
    """
    Returns list of dicts:
        { 'rel_path': 'Celeb-real/id1_0007.mp4',
          'abs_path': '...full path...',
          'label': 0 or 1,
          'subject': 'id1' }
    """
    videos = []

    for folder in real_folders:
        folder_path = os.path.join(celeb_root, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder not found: {folder_path}")
            continue
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".mp4"):
                continue
            videos.append({
                "rel_path": f"{folder}/{f}",
                "abs_path": os.path.join(folder_path, f),
                "label":    0,   # real in pipeline convention
                "subject":  extract_subject_id(f)
            })

    for folder in fake_folders:
        folder_path = os.path.join(celeb_root, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder not found: {folder_path}")
            continue
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".mp4"):
                continue
            videos.append({
                "rel_path": f"{folder}/{f}",
                "abs_path": os.path.join(folder_path, f),
                "label":    1,   # fake in pipeline convention
                "subject":  extract_subject_id(f)
            })

    return videos


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(SEED)

    print("=" * 60)
    print("CelebDF-v2 Dataset Preparation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load official test list
    # ------------------------------------------------------------------
    print("\n[1] Loading test list...")
    test_set = load_test_list(TEST_LIST_PATH)
    print(f"    Test videos: {len(test_set)}")

    real_test = sum(1 for v in test_set.values() if v == 0)
    fake_test = sum(1 for v in test_set.values() if v == 1)
    print(f"    Real: {real_test}  Fake: {fake_test}")

    # ------------------------------------------------------------------
    # 2. Collect all videos from disk
    # ------------------------------------------------------------------
    print("\n[2] Scanning disk...")
    all_videos = collect_all_videos(CELEB_ROOT, REAL_FOLDERS, FAKE_FOLDERS)
    print(f"    Total videos found: {len(all_videos)}")

    # ------------------------------------------------------------------
    # 3. Separate test vs remaining
    # ------------------------------------------------------------------
    print("\n[3] Separating test / remaining...")

    test_videos      = []
    remaining_videos = []

    missing_from_disk = []

    for v in all_videos:
        if v["rel_path"] in test_set:
            # Verify label consistency
            expected_label = test_set[v["rel_path"]]
            if v["label"] != expected_label:
                print(f"[WARN] Label mismatch for {v['rel_path']}: "
                      f"disk={v['label']} txt={expected_label}")
            test_videos.append(v)
        else:
            remaining_videos.append(v)

    # Check for test videos listed in txt but missing on disk
    disk_rel_paths = {v["rel_path"] for v in all_videos}
    for rel_path in test_set:
        if rel_path not in disk_rel_paths:
            missing_from_disk.append(rel_path)

    if missing_from_disk:
        print(f"\n[WARN] {len(missing_from_disk)} test videos listed in txt "
              f"but NOT found on disk:")
        for p in missing_from_disk[:10]:
            print(f"    {p}")
        if len(missing_from_disk) > 10:
            print(f"    ... and {len(missing_from_disk) - 10} more")

    print(f"    Test set    : {len(test_videos)} videos")
    print(f"    Remaining   : {len(remaining_videos)} videos")

    # ------------------------------------------------------------------
    # 4. Subject-level train/val split on remaining videos
    # ------------------------------------------------------------------
    print("\n[4] Subject-level train/val split...")

    # Group remaining by subject
    subject_to_videos = defaultdict(list)
    for v in remaining_videos:
        subject_to_videos[v["subject"]].append(v)

    all_subjects = sorted(subject_to_videos.keys())
    random.shuffle(all_subjects)

    n_train_subjects = int(len(all_subjects) * TRAIN_RATIO)
    train_subjects   = set(all_subjects[:n_train_subjects])
    val_subjects     = set(all_subjects[n_train_subjects:])

    train_videos = []
    val_videos   = []

    for v in remaining_videos:
        if v["subject"] in train_subjects:
            train_videos.append(v)
        else:
            val_videos.append(v)

    print(f"    Subjects total : {len(all_subjects)}")
    print(f"    Train subjects : {len(train_subjects)}")
    print(f"    Val subjects   : {len(val_subjects)}")
    print(f"    Train videos   : {len(train_videos)}")
    print(f"    Val videos     : {len(val_videos)}")

    # ------------------------------------------------------------------
    # 5. Class balance report
    # ------------------------------------------------------------------
    print("\n[5] Class balance:")

    for split_name, split_vids in [
        ("train", train_videos),
        ("val",   val_videos),
        ("test",  test_videos)
    ]:
        real = sum(1 for v in split_vids if v["label"] == 0)
        fake = sum(1 for v in split_vids if v["label"] == 1)
        ratio = fake / real if real > 0 else float("inf")
        print(f"    {split_name:5s} → real: {real:4d}  fake: {fake:4d}  "
              f"fake/real: {ratio:.2f}")

    # ------------------------------------------------------------------
    # 6. Create output folder structure
    # ------------------------------------------------------------------
    print("\n[6] Creating output folders...")

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            os.makedirs(
                os.path.join(OUTPUT_ROOT, split, cls),
                exist_ok=True
            )

    # ------------------------------------------------------------------
    # 7. Transfer files
    # ------------------------------------------------------------------
    print("\n[7] Transferring files (hardlink → symlink → copy)...")

    label_to_cls = {0: "real", 1: "fake"}

    stats = {"train": {"real": 0, "fake": 0},
             "val":   {"real": 0, "fake": 0},
             "test":  {"real": 0, "fake": 0}}

    for split_name, split_vids in [
        ("train", train_videos),
        ("val",   val_videos),
        ("test",  test_videos)
    ]:
        for v in split_vids:
            cls      = label_to_cls[v["label"]]
            filename = os.path.basename(v["abs_path"])
            dst      = os.path.join(OUTPUT_ROOT, split_name, cls, filename)

            transfer_file(v["abs_path"], dst)
            stats[split_name][cls] += 1

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        real = stats[split]["real"]
        fake = stats[split]["fake"]
        total = real + fake
        print(f"\n{split.upper()} SET  ({total} videos)")
        print(f"  Real : {real}")
        print(f"  Fake : {fake}")
        print(f"  Fake/Real ratio: {fake/real:.2f}" if real > 0 else "  No real videos")

    print(f"\nOutput root: {OUTPUT_ROOT}")
    print("\n✅ Done. Now run cache_face_3d.py pointing to OUTPUT_ROOT")



if __name__ == "__main__":
    main()
# output is shown in the comment below
#============================================================
#CelebDF-v2 Dataset Preparation
#============================================================

#[1] Loading test list...
#    Test videos: 518
#    Real: 178  Fake: 340

#[2] Scanning disk...
#    Total videos found: 6529

#[3] Separating test / remaining...
#    Test set    : 518 videos
#    Remaining   : 6011 videos

#[4] Subject-level train/val split...
#    Subjects total : 289
#    Train subjects : 231
#    Val subjects   : 58
#    Train videos   : 4643
#    Val videos     : 1368

#[5] Class balance:
#    train → real:  557  fake: 4086  fake/real: 7.34
#    val   → real:  155  fake: 1213  fake/real: 7.83
#    test  → real:  178  fake:  340  fake/real: 1.91

#[6] Creating output folders...

#[7] Transferring files (hardlink → symlink → copy)...

#============================================================
#FINAL DATASET SUMMARY
#============================================================

#TRAIN SET  (4643 videos)
#  Real : 557
#  Fake : 4086
#  Fake/Real ratio: 7.34

#VAL SET  (1368 videos)
#  Real : 155
#  Fake : 1213
#  Fake/Real ratio: 7.83

#TEST SET  (518 videos)
#  Real : 178
#  Fake : 340
#  Fake/Real ratio: 1.91

#Output root: F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF

#✅ Done. Now run cache_face_3d.py pointing to OUTPUT_ROOT
