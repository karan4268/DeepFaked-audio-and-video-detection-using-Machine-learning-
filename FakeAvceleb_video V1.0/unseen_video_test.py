# Unseen Video Test Script
# Description:
#   - Tests the video model on unseen video files.
#   - Extracts frames, applies MTCNN for face detection, and runs inference.
#   - Multi-pass inference (with and without MTCNN) is used and averaged.
#   - Supports batch testing on multiple videos in a folder.

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from video_model import VideoModel
import os
import csv

# ================= CONFIG ================= #
VIDEO_FOLDER = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\My test clips"
MODEL_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\video_model_best_3D_CNN-ResNet18.pth"

NUM_FRAMES = 24      # Number of frames to extract from each video
NUM_CLIPS = 3        # Number of temporal clips for stable prediction
IMG_SIZE = 224       # Image size (224 x 224)
THRESHOLD = 0.5      # Decision threshold for classification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================================== #

# ImageNet normalization for input frames
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# ------------------------------------------------ #
def center_crop_fallback(frame):
    """
    Fallback cropping when face detection fails.
    Crops the center square and resizes to IMG_SIZE.
    """
    h, w, _ = frame.shape
    min_dim = min(h, w)
    y1 = (h - min_dim) // 2
    x1 = (w - min_dim) // 2
    crop = frame[y1:y1 + min_dim, x1:x1 + min_dim]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return crop

# ------------------------------------------------ #
def sample_frames(video_path, num_frames, mtcnn=None, start_ratio=0.0):
    """
    Samples frames from a video, applies face detection if MTCNN is provided.
    Returns a tensor of shape [C, T, H, W].
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise RuntimeError("❌ Empty or unreadable video")

    start_frame = int(start_ratio * total_frames)
    end_frame = min(start_frame + num_frames * 5, total_frames - 1)
    frame_ids = np.linspace(start_frame, end_frame, num_frames, dtype=int)
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

    # Pad if not enough frames
    while len(collected) < num_frames:
        collected.append(collected[-1])

    clip = torch.stack(collected[:num_frames])   # [T, C, H, W]
    clip = clip.permute(1, 0, 2, 3)              # [C, T, H, W]
    return clip

# ------------------------------------------------ #
def run_inference(video_path, use_mtcnn):
    """
    Runs inference on a video using the model.
    If use_mtcnn is True, uses face detection; otherwise, uses center crop.
    Returns the average fake probability across NUM_CLIPS.
    """
    mtcnn = None
    if use_mtcnn:
        mtcnn = MTCNN(
            image_size=IMG_SIZE,
            margin=20,
            keep_all=True,
            device=DEVICE
        )

    clip_scores = []
    for i in range(NUM_CLIPS):
        start_ratio = i / NUM_CLIPS
        frames = sample_frames(
            video_path,
            NUM_FRAMES,
            mtcnn=mtcnn,
            start_ratio=start_ratio
        )
        frames = frames.unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(frames)
            probs = F.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()
        clip_scores.append(fake_prob)

    return np.mean(clip_scores)

# ---------------- LOAD MODEL ---------------- #
model = VideoModel().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# -------------------------------------------- #

# ----------- FIND VIDEO FILES --------------- #
video_files = [
    os.path.join(VIDEO_FOLDER, f)
    for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]
print(f"[INFO] Found {len(video_files)} videos for testing")
# -------------------------------------------- #

results = []

# ---------- mtcnn Fusion FUNCTION ----------
def fuse_predictions(face_prob, frame_prob):

    # if face model is confident → trust it
    if abs(face_prob - 0.5) > 0.20:
        final_score = face_prob
    else:
        # otherwise combine both
        final_score = 0.6 * frame_prob + 0.4 * face_prob

    return final_score


# ---------- CLASSIFICATION FUNCTION ----------
def classify_score(score, low=0.45, high=0.55):

    if score >= high:
        return "FAKE"
    elif score <= low:
        return "REAL"
    else:
        return "UNCERTAIN"


# ---------------- RUN INFERENCE ---------------- #
for VIDEO_PATH in video_files:

    # Run inference with face detection
    fake_prob_mtcnn = run_inference(VIDEO_PATH, use_mtcnn=True)

    # Run inference without face detection
    fake_prob_full  = run_inference(VIDEO_PATH, use_mtcnn=False)

    # Fuse both predictions
    final_score = fuse_predictions(fake_prob_mtcnn, fake_prob_full)

    # Convert score to label
    prediction = classify_score(final_score)

    print("\n========== VIDEO PREDICTION ==========")
    print(f"Video          : {VIDEO_PATH}")
    print(f"MTCNN On Prob  : {fake_prob_mtcnn:.4f}")
    print(f"MTCNN OFF Prob : {fake_prob_full:.4f}")
    print(f"Final Score    : {final_score:.4f}")
    print(f"Prediction     : {prediction}")
    print("=====================================\n")

    results.append([
        VIDEO_PATH,
        fake_prob_mtcnn,
        fake_prob_full,
        final_score,
        prediction
    ])

# ----------------------------------------------- #


# -------- SAVE RESULTS TO CSV -------- #
csv_path = "unseen_video_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "mtcnn_score", "frame_score", "final_score", "prediction"])
    writer.writerows(results)

print(f"[INFO] Results saved to: {csv_path}")