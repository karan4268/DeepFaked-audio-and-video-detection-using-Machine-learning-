import os
import hashlib

ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video"
SPLIT = "train"

video_root = os.path.join(ROOT, SPLIT)
cache_root = os.path.join(ROOT, "cached_faces", SPLIT)

missing = []

for cls in ["real", "fake"]:
    cls_dir = os.path.join(video_root, cls)
    for root, _, files in os.walk(cls_dir):
        for f in files:
            if not f.endswith(".mp4"):
                continue

            video_path = os.path.join(root, f)
            rel = os.path.relpath(video_path, video_root)
            h = hashlib.md5(rel.encode()).hexdigest()
            cache_dir = os.path.join(cache_root, h)

            if not os.path.isdir(cache_dir):
                missing.append(video_path)

print(f"\nMissing cache for {len(missing)} videos\n")
for v in missing[:10]:
    print(v)
