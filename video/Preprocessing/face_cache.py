# face_cache.py
# ------------------------------------------------------------
# Optimized Face Cache for 3D CNN
# - Batched face detection (faster)
# - Temporal consistency (stable face box across frames)
# - Robust caching + validation
# - Parallel processing
# ------------------------------------------------------------
#
# CHANGES FROM ORIGINAL:
#   [NO CHANGE] Cache generation stores raw pixel PNGs with NO normalization.
#   This is intentionally correct — normalization happens at DataLoader time
#   using fixed ImageNet constants (video_preprocessing.py).  Do not add
#   normalization here; it would bake the transform into the cache files and
#   make them unusable if you ever change the normalization strategy.
#
#  currently is used for CelebDF but can be adapted to other datasets with similar structure by changing ROOT_DIR and SPLITS.
# ------------------------------------------------------------

import os
import cv2
import torch
import hashlib
import shutil
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23" # change root to specific dataset
SPLITS   = ["train", "val", "test"]

FRAMES_PER_VIDEO    = 24
IMG_SIZE            = 224
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

MIN_FACE_AREA_RATIO = 0.01
MIN_CONFIDENCE      = 0.90
BATCH_SIZE          = 32      # better GPU utilization

# ------------------------------------------------------------
detector = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=True,
    device=DEVICE
)

# ------------------------------------------------------------
def sample_frame_ids(total_frames, n_frames):
    return np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

# ------------------------------------------------------------
def video_cache_dir(video_path, root_dir, split):
    rel = os.path.relpath(video_path, os.path.join(root_dir, split))
    h   = hashlib.md5(rel.encode("utf-8")).hexdigest()
    return os.path.join(root_dir, "cached_faces", split, h)

# ------------------------------------------------------------
def center_crop(frame):
    """Square-crop and resize to IMG_SIZE.  Returns uint8 RGB ndarray."""
    h, w, _ = frame.shape
    min_dim  = min(h, w)
    y1 = (h - min_dim) // 2
    x1 = (w - min_dim) // 2
    crop = frame[y1:y1 + min_dim, x1:x1 + min_dim]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

# ------------------------------------------------------------
def is_cache_valid(out_dir):
    if not os.path.isdir(out_dir):
        return False
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    return len(files) == FRAMES_PER_VIDEO

# ------------------------------------------------------------
def select_face(boxes, probs, frame_shape, prev_box=None):
    if boxes is None or probs is None:
        return None

    h, w, _ = frame_shape
    best       = None
    best_score = -1

    for box, prob in zip(boxes, probs):
        if prob < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)

        if area < MIN_FACE_AREA_RATIO * (h * w):
            continue

        score = area

        if prev_box is not None:
            iou    = compute_iou(box, prev_box)
            score += iou * 1000

        if score > best_score:
            best_score = score
            best       = (x1, y1, x2, y2)

    return best

# ------------------------------------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)

# ------------------------------------------------------------
def cache_video(video_path, root_dir, split):
    out_dir = video_cache_dir(video_path, root_dir, split)

    if is_cache_valid(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    frame_ids = set(sample_frame_ids(total_frames, FRAMES_PER_VIDEO))

    frames = []
    current_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_id in frame_ids:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) == FRAMES_PER_VIDEO:
                break

        current_id += 1

    cap.release()

    if not frames:
        return

    # Batched MTCNN detection
    boxes_batch = []
    probs_batch = []

    for i in range(0, len(frames), BATCH_SIZE):
        batch     = frames[i:i + BATCH_SIZE]
        pil_batch = [Image.fromarray(f) for f in batch]
        b, p      = detector.detect(pil_batch)
        boxes_batch.extend(b)
        probs_batch.extend(p)

    saved    = 0
    prev_box = None

    for i, frame in enumerate(frames):
        boxes    = boxes_batch[i]
        probs    = probs_batch[i]
        face_box = select_face(boxes, probs, frame.shape, prev_box)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Store raw uint8 pixels — NO normalization here.
                # Normalization is applied at DataLoader time with fixed
                # ImageNet constants so it can be changed without re-caching.
                face     = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                prev_box = face_box
            else:
                face = center_crop(frame)
        else:
            face = center_crop(frame)

        Image.fromarray(face).save(
            os.path.join(out_dir, f"{saved:03d}.png")
        )
        saved += 1

    # Pad to exactly FRAMES_PER_VIDEO if fewer frames were detected
    if saved > 0:
        last = os.path.join(out_dir, f"{saved - 1:03d}.png")
        for i in range(saved, FRAMES_PER_VIDEO):
            shutil.copy(last, os.path.join(out_dir, f"{i:03d}.png"))

# ------------------------------------------------------------
def process_split(split):
    video_root = os.path.join(ROOT_DIR, split)

    paths = []
    for cls in ["real", "fake"]:
        cls_dir = os.path.join(video_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.endswith(".mp4"):
                    paths.append(os.path.join(root, f))

    print(f"[INFO] {split}: {len(paths)} videos")

    with ThreadPoolExecutor(max_workers=6) as executor: # adjust based on your CPU cores and cpu/gpu balance
        list(tqdm(
            executor.map(lambda p: cache_video(p, ROOT_DIR, split), paths),
            total=len(paths)
        ))

# ------------------------------------------------------------
if __name__ == "__main__":
    print("● Building optimized 3D CNN face cache...")

    for split in SPLITS:
        process_split(split)

    print("\n✅ Cache completed successfully.")

# for CelebDB
#● Building optimized 3D CNN face cache...
#[INFO] train: 4643 videos
#100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4643/#4643 [1:16:27<00:00,  1.01it/s]
#[INFO] val: 1368 videos
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1368/1368 [21:09<00:00,  1.08it/s]
#[INFO] test: 518 videos
#100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 518/518 [08:01<00:00,  1.08it/s]

# for FF++C23
#● Building optimized 3D CNN face cache...
#[INFO] train: 4200 videos
#100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4200/4200 [2:24:53<00:00,  2.07s/it]
#[INFO] val: 900 videos
#100%|████████████████████████████████ ████████| 900/900 [29:15<00:00,  1.95s/it]
#[INFO] test: 900 videos
#100%| ████████| 900/900 [38:03<00:00,  2.54s/it]

#✅ Cache completed successfully.