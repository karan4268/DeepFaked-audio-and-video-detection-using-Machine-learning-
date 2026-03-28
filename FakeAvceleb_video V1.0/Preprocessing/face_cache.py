# ------------------------------------------------------------
# MTech Project: DeepFaked Video and Audio Analyzer
# Module: Video Preprocessing - Face_Cache.py
# Face Cache Creation for FakeAVCeleb Dataset
# This script processes all videos in the FakeAVCeleb dataset, extracts faces using MTCNN
# =======================================================================================================================================
# Implements the following:
# 1. Iterates through all videos in the dataset, organized by class folders.
# 2. For each video, samples a fixed number of frames = 24
# 3. Uses MTCNN to detect faces in each sampled frame, selects the largest valid face, and saves it as a JPEG. If no valid face is detected, applies a center crop as fallback.
# 4. Saves the extracted faces in a structured cache directory with a unique hash-based name derived from the video path.
# 5. Creates a label.txt file for each cached video containing both primary (4-class) and binary labels.
# 6. saves a bbox.json file with the bounding box coordinates for each extracted.
# 7. After processing all videos, creates a dataset_split.json file that organizes the cached videos into train/val/test splits with      their corresponding labels.

# primary_label would be usefull for fusion layer when audio model is also used. For video-only model, binary_label should be used.
# -------------------------------------------------------------
# Primary labels (4-class):
#   0: RealVideo-RealAudio (RR)
#   1: FakeVideo-RealAudio (FR)
#   2: RealVideo-FakeAudio (RF)
#   3: FakeVideo-FakeAudio (FF)
# ------------------------------------------------------------
# Binary labels (for video-only model):
#   0: Real (RR + RF)   
#   1: Fake (FR + FF) 
# ------------------------------------------------------------
# VIDEO_BINARY_MAP = 
#    0: 0,  # RR -> Real label.txt primary_label=0, binary_label=0
#    1: 1,  # FR -> Fake label.txt primary_label=1, binary_label=1
#    2: 0,  # RF -> Real label.txt primary_label=2, binary_label=0 (treated as real for binary)
#    3: 1   # FF -> Fake label.txt primary_label=3, binary_label=1
# ------------------------------------------------------------
# ========================================================================================================================================
import os
import cv2
import json
import torch
import hashlib
import shutil
import random
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm # for progress bars

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ROOT_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2"
FRAMES_PER_VIDEO = 24
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_FACE_AREA_RATIO = 0.01
MIN_CONFIDENCE = 0.85
SAVE_BBOX_JSON = True
BINARY_CLASSIFICATION = True  # 0=real,1=fake (class2 treated as real)
VAL_FRACTION = 0.2            # fraction for val/test split
MAX_VIDEOS_PER_CLASS = None   # None = cache all

CLASS_MAP_4 = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "RealVideo-FakeAudio": 2,  # binary treated as real
    "FakeVideo-FakeAudio": 3,
}

# ------------------------------------------------------------
# MTCNN Face Detector
# ------------------------------------------------------------
detector = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=True,
    device=DEVICE
)

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def sample_frame_ids(total_frames, n_frames):
    return np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

def video_cache_dir(video_path, root_dir):
    rel = os.path.relpath(video_path, root_dir)
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()
    return os.path.join(root_dir, "cached_faces", h)

def center_crop(frame, size=IMG_SIZE):
    h, w, _ = frame.shape
    min_dim = min(h, w)
    y1 = (h - min_dim) // 2
    x1 = (w - min_dim) // 2
    crop = frame[y1:y1 + min_dim, x1:x1 + min_dim]
    return cv2.resize(crop, (size, size))

def select_largest_face(boxes, probs, w, h):
    valid_faces = []
    for box, prob in zip(boxes, probs):
        if prob < MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_FACE_AREA_RATIO * (w * h):
            valid_faces.append((area, (x1, y1, x2, y2)))
    if not valid_faces:
        return None
    valid_faces.sort(reverse=True)
    return valid_faces[0][1]

def infer_labels(video_path):
    parts = video_path.split(os.sep)
    for p in parts:
        if p in CLASS_MAP_4:
            primary_label = CLASS_MAP_4[p]
            if BINARY_CLASSIFICATION:
                binary_label = 0 if p in ["RealVideo-RealAudio","RealVideo-FakeAudio"] else 1
            else:
                binary_label = primary_label
            return primary_label, binary_label
    raise RuntimeError(f"Could not infer label for {video_path}")

# ------------------------------------------------------------
# CACHE VIDEO
# ------------------------------------------------------------
def cache_video(video_path, root_dir):
    primary_label, binary_label = infer_labels(video_path)
    out_dir = video_cache_dir(video_path, root_dir)

    # Resume-safe
    if os.path.isdir(out_dir):
        jpgs = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        if len(jpgs) >= FRAMES_PER_VIDEO:
            return out_dir, primary_label, binary_label

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, primary_label, binary_label

    frame_ids = sample_frame_ids(total_frames, FRAMES_PER_VIDEO)
    target_ids = set(frame_ids)
    saved = 0
    current_frame = 0
    bbox_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in target_ids:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            boxes, probs = detector.detect(frame_rgb)
            selected_box = None
            face = None

            if boxes is not None and probs is not None:
                selected_box = select_largest_face(boxes, probs, w, h)
                if selected_box is not None:
                    x1, y1, x2, y2 = selected_box
                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        face = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            if face is None:
                face = center_crop(frame_rgb)

            Image.fromarray(face).save(os.path.join(out_dir, f"{saved:03d}.jpg"), format="JPEG", quality=95)
            if SAVE_BBOX_JSON:
                bbox_data[f"{saved:03d}"] = selected_box if selected_box else "center_crop"

            saved += 1
            if saved == FRAMES_PER_VIDEO:
                break
        current_frame += 1

    cap.release()

    if saved == 0:
        shutil.rmtree(out_dir)
        return None, primary_label, binary_label

    # Padding
    last_frame_path = os.path.join(out_dir, f"{saved - 1:03d}.jpg")
    for i in range(saved, FRAMES_PER_VIDEO):
        shutil.copy(last_frame_path, os.path.join(out_dir, f"{i:03d}.jpg"))

    # Save labels
    with open(os.path.join(out_dir, "label.txt"), "w") as f:
        f.write(f"{primary_label},{binary_label}")

    if SAVE_BBOX_JSON:
        with open(os.path.join(out_dir, "bbox.json"), "w") as f:
            json.dump(bbox_data, f, indent=2)

    return out_dir, primary_label, binary_label

# ------------------------------------------------------------
# PROCESS AND SPLIT FULL CACHE (IDENTITY BASED)
# ------------------------------------------------------------
def process_fakeavceleb(root_dir):
    class_folders = ["RealVideo-RealAudio","FakeVideo-RealAudio","RealVideo-FakeAudio","FakeVideo-FakeAudio"]
    split_json = {"train_full_pool": [], "val": [], "test": []}
    total_videos = 0

    for class_name in class_folders:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"📁 Processing {class_name}")

        # ------------------------------------------------------------
        # Collect identity folders (id000xx level)
        # ------------------------------------------------------------
        identity_dirs = []

        for root, dirs, files in os.walk(class_path):
            if any(f.lower().endswith(".mp4") for f in files):
                identity_dirs.append(root)

        random.shuffle(identity_dirs)

        # Optional limit
        if MAX_VIDEOS_PER_CLASS is not None:
            identity_dirs = identity_dirs[:MAX_VIDEOS_PER_CLASS]

        # ------------------------------------------------------------
        # Split identities (NOT videos)
        # ------------------------------------------------------------
        val_count = int(len(identity_dirs) * VAL_FRACTION)
        test_count = val_count
        train_count = len(identity_dirs) - val_count - test_count

        train_ids = identity_dirs[:train_count]
        val_ids   = identity_dirs[train_count:train_count+val_count]
        test_ids  = identity_dirs[train_count+val_count:]

        # ------------------------------------------------------------
        # Collect videos from each identity split
        # ------------------------------------------------------------
        def collect_videos(id_list):
            vids = []
            for id_dir in id_list:
                for f in os.listdir(id_dir):
                    if f.lower().endswith(".mp4"):
                        vids.append(os.path.join(id_dir, f))
            return vids

        train_videos = collect_videos(train_ids)
        val_videos   = collect_videos(val_ids)
        test_videos  = collect_videos(test_ids)

        # ------------------------------------------------------------
        # Cache videos
        # ------------------------------------------------------------
        for vid_path in tqdm(train_videos, desc=f"{class_name} Train", leave=False):
            cached_dir, primary_label, binary_label = cache_video(vid_path, root_dir)
            if cached_dir:
                split_json["train_full_pool"].append({
                    "path": cached_dir,
                    "primary_label": primary_label,
                    "binary_label": binary_label
                })

        for vid_path in tqdm(val_videos, desc=f"{class_name} Val", leave=False):
            cached_dir, primary_label, binary_label = cache_video(vid_path, root_dir)
            if cached_dir:
                split_json["val"].append({
                    "path": cached_dir,
                    "primary_label": primary_label,
                    "binary_label": binary_label
                })

        for vid_path in tqdm(test_videos, desc=f"{class_name} Test", leave=False):
            cached_dir, primary_label, binary_label = cache_video(vid_path, root_dir)
            if cached_dir:
                split_json["test"].append({
                    "path": cached_dir,
                    "primary_label": primary_label,
                    "binary_label": binary_label
                })

        total_videos += len(train_videos) + len(val_videos) + len(test_videos)

        print(f"✅ Cached {len(train_videos)+len(val_videos)+len(test_videos)} videos for {class_name}")

    # Save JSON
    with open(os.path.join(root_dir, "dataset_split.json"), "w") as f:
        json.dump(split_json, f, indent=2)

    print(f"\n✅ FakeAVCeleb full cache + dual-label JSON completed")
    print(f"Total videos cached: {total_videos}")
# ------------------------------------------------------------
if __name__ == "__main__":
    process_fakeavceleb(ROOT_DIR)

# ----------------output date:-12-03-2026---------------------- (id based split)
    # 📁 Processing RealVideo-RealAudio
#✅ Cached 500 videos for RealVideo-RealAudio
#📁 Processing FakeVideo-RealAudio
#✅ Cached 9709 videos for FakeVideo-RealAudio
#📁 Processing RealVideo-FakeAudio
#✅ Cached 500 videos for RealVideo-FakeAudio
#📁 Processing FakeVideo-FakeAudio
#✅ Cached 10851 videos for FakeVideo-FakeAudio

#✅ FakeAVCeleb full cache + dual-label JSON completed
#Total videos cached: 21560