# =============================================================================
# convert_cache_to_npy.py
# Converts FakeAVCeleb PNG frame cache → single .npy per video
#
# Before: D:\FakeAVCache\Video\<md5hash>\000.png ... 023.png
# After:  D:\FakeAVCache\Video\<md5hash>\frames.npy  shape: (24, 224, 224, 3) uint8
#
# The PNG folders are kept intact — npy sits alongside them.
# Delete PNGs manually after verifying if you want to free space.
#
# Usage:
#   python convert_cache_to_npy.py
#   python convert_cache_to_npy.py --cache_root D:\FakeAVCache\Video
#   python convert_cache_to_npy.py --workers 3 --delete_pngs
# =============================================================================

import os
import argparse
import json
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_CACHE_ROOT = r"D:\FakeAVCache\Video"
FRAMES_PER_VIDEO   = 24
NPY_FILENAME       = "frames.npy"


# =============================================================================
# CONVERT ONE VIDEO DIRECTORY
# =============================================================================

def convert_one(args):
    frame_dir, delete_pngs = args

    npy_path = os.path.join(frame_dir, NPY_FILENAME)

    # Skip if already converted
    if os.path.isfile(npy_path):
        return "skip"

    all_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0])
    )

    if len(all_files) == 0:
        return "empty"

    # Pad if incomplete
    if len(all_files) < FRAMES_PER_VIDEO:
        all_files += [all_files[-1]] * (FRAMES_PER_VIDEO - len(all_files))

    file_slice = all_files[:FRAMES_PER_VIDEO]

    frames = []
    for fname in file_slice:
        with Image.open(os.path.join(frame_dir, fname)) as img:
            frames.append(np.array(img.convert("RGB"), dtype=np.uint8))

    # Stack → (24, 224, 224, 3) uint8
    # Stored as uint8 to keep file size small (~3.6MB vs ~14MB float32)
    clip = np.stack(frames, axis=0)
    np.save(npy_path, clip)

    # Optionally delete PNGs to free space
    if delete_pngs:
        for fname in all_files:
            png_path = os.path.join(frame_dir, fname)
            if os.path.isfile(png_path):
                os.remove(png_path)

    return "ok"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert PNG cache to NPY")
    parser.add_argument("--cache_root",   default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers",      type=int, default=3,
                        help="Parallel conversion workers (match your CPU cores)")
    parser.add_argument("--delete_pngs",  action="store_false",
                        help="Delete PNG files after successful NPY conversion")
    args = parser.parse_args()

    # Load index to get all hash dirs
    index_path = os.path.join(args.cache_root, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with open(index_path, "r") as f:
        samples = json.load(f)

    frame_dirs = [
        os.path.join(args.cache_root, s["file"])
        for s in samples
    ]

    print(f"[INFO] Total videos to convert : {len(frame_dirs)}")
    print(f"[INFO] Workers                 : {args.workers}")
    print(f"[INFO] Delete PNGs after       : {args.delete_pngs}")
    if args.delete_pngs:
        print(f"[WARN] PNG deletion is ON — this is irreversible")

    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}

    work = [(d, args.delete_pngs) for d in frame_dirs]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, w): w for w in work}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                stats[result] = stats.get(result, 0) + 1
            except Exception as e:
                stats["error"] += 1

    print(f"\n===== Conversion Complete =====")
    print(f"  Converted : {stats['ok']}")
    print(f"  Skipped   : {stats['skip']}  (already had frames.npy)")
    print(f"  Empty     : {stats['empty']}")
    print(f"  Errors    : {stats['error']}")

    # Estimate space saved
    if args.delete_pngs:
        saved_gb = stats["ok"] * FRAMES_PER_VIDEO * (224 * 224 * 3) / (1024**3)
        print(f"\n[INFO] Approx space freed : {saved_gb:.1f} GB")
    else:
        npy_size_gb  = len(frame_dirs) * 24 * 224 * 224 * 3 / (1024**3)
        print(f"\n[INFO] Approx NPY cache size added : {npy_size_gb:.1f} GB")
        print(f"[INFO] Run with --delete_pngs to free the PNG space after verifying")


if __name__ == "__main__":
    main()