import time
import torch
from torch.utils.data import DataLoader
from Preprocessing.video_preprocessing import DeepfakedDataset

device = "cuda"
ds = DeepfakedDataset(
    root_dir=r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF",
    split="train", frames_per_video=24, device=device
)
loader = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0)

# time just the dataloader (CPU side — no GPU)
t = time.time()
frames, labels = next(iter(loader))
print(f"DataLoader (CPU): {(time.time()-t)*1000:.1f}ms")

# time GPU transfer
t = time.time()
frames = frames.to(device, non_blocking=True)
torch.cuda.synchronize()
print(f"GPU transfer: {(time.time()-t)*1000:.1f}ms")

# time forward pass
from video_model import VideoModel
model = VideoModel().to(device)
model.eval()
t = time.time()
with torch.cuda.amp.autocast():
    out = model(frames)
torch.cuda.synchronize()
print(f"Forward pass: {(time.time()-t)*1000:.1f}ms")

# time backward pass
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(out, labels.to(device))
t = time.time()
loss.backward()
torch.cuda.synchronize()
print(f"Backward pass: {(time.time()-t)*1000:.1f}ms")