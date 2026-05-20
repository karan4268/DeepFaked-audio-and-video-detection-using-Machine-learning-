# Unseen Video Test Script
# - Description: This script tests the video model on an unseen video file. It extracts frames, applies MTCNN for face detection, and runs inference to determine if the video is real or fake.
# - added multi pass inference (with and without MTCNN) and averages the results for a more robust prediction.
# 

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from video_model import VideoModel

# ================= CONFIG ================= #
VIDEO_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\inference test clips\celeb DB\test\real\id25_0005.mp4"
MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Trained and Tested Models\ver2 with cache faces and MtCNN\video_model_best.pth"

NUM_FRAMES = 24 # 24 frames are extracted form the video
IMG_SIZE = 224 # 224 x 224 Img size
THRESHOLD = 0.5 # 0.5 is default Optimization Required

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================================== #

# ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# ------------------------------------------------ #
def center_crop_fallback(frame):
    h, w, _ = frame.shape
    min_dim = min(h, w)
    y1 = (h - min_dim) // 2
    x1 = (w - min_dim) // 2
    crop = frame[y1:y1 + min_dim, x1:x1 + min_dim]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return crop
# ------------------------------------------------ #

def sample_frames(video_path, num_frames, mtcnn=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise RuntimeError("❌ Empty or unreadable video")

    frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    collected = []

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------- FACE PATH --------- #
        if mtcnn is not None:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = np.argmax(areas)
                face = mtcnn.extract(frame, boxes[idx:idx + 1], save_path=None)

                if face is not None:
                    face = normalize(face[0])
                    collected.append(face)
                    continue

        # --------- FALLBACK --------- #
        crop = center_crop_fallback(frame)
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop = normalize(crop)
        collected.append(crop)

    cap.release()

    if len(collected) == 0:
        raise RuntimeError("❌ No usable frames extracted")

    while len(collected) < num_frames:
        collected.append(collected[-1])

    return torch.stack(collected[:num_frames])
# ------------------------------------------------ #

def run_inference(video_path, use_mtcnn):
    mtcnn = None
    if use_mtcnn:
        mtcnn = MTCNN(
            image_size=IMG_SIZE,
            margin=20,
            keep_all=True,
            device=DEVICE
        )

    frames = sample_frames(video_path, NUM_FRAMES, mtcnn)
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(frames)
        probs = F.softmax(logits, dim=1)
        fake_prob = probs[0, 1].item()

    return fake_prob

# ---------------- LOAD MODEL ---------------- #
model = VideoModel().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# -------------------------------------------- #

# ---------------- RUN BOTH PASSES ---------------- #
fake_prob_mtcnn = run_inference(VIDEO_PATH, use_mtcnn=True)
fake_prob_full  = run_inference(VIDEO_PATH, use_mtcnn=False)

final_score = 0.7 * fake_prob_mtcnn + 0.3 * fake_prob_full #

# ------------------------------------------------- #

# ---------------- DECISION ---------------- #
if final_score >= THRESHOLD:
    prediction = "FAKE"
else:
    prediction = "REAL"
# ------------------------------------------ #

print("\n========== VIDEO PREDICTION ==========")
print(f"Video          : {VIDEO_PATH}")
print(f"MTCNN On Prob  : {fake_prob_mtcnn:.4f}")
print(f"MTCNN OFF Prob : {fake_prob_full:.4f}")
print(f"Final Score    : {final_score:.4f}")
print(f"Threshold      : {THRESHOLD}")
print(f"Prediction     : {prediction}")
print("=====================================\n")
