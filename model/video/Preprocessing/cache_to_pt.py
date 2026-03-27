# convert_cached_faces_to_pt.py

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================
SOURCE_DIR = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2\cached_faces"
DEST_DIR   = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\FakeAVceleb\FakeAVCeleb_v1.2\video_tensors"

FRAMES_PER_VIDEO = 24
IMG_SIZE = 224

os.makedirs(DEST_DIR, exist_ok=True)

# =====================================================
# Transform (MUST match training config)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# Conversion
# =====================================================
video_folders = os.listdir(SOURCE_DIR)

for vid in tqdm(video_folders):

    vid_path = os.path.join(SOURCE_DIR, vid)
    if not os.path.isdir(vid_path):
        continue

    frame_files = sorted(os.listdir(vid_path))

    # Ensure consistent frame count
    if len(frame_files) < FRAMES_PER_VIDEO:
        continue  # skip short clips

    frame_files = frame_files[:FRAMES_PER_VIDEO]

    frames = []

    for frame_name in frame_files:
        img_path = os.path.join(vid_path, frame_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            frames.append(img)
        except:
            frames = []
            break

    if len(frames) != FRAMES_PER_VIDEO:
        continue

    clip = torch.stack(frames)  # (T, C, H, W)

    save_path = os.path.join(DEST_DIR, vid + ".pt")
    torch.save(clip, save_path)

print("✅ All videos converted to .pt tensors")