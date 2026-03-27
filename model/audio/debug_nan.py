# debug_batch_nan.py
import os, json, torch, numpy as np
from torch.utils.data import DataLoader
from audio_model import AudioResNet18
from data_loader import ASVSpoofDataset, collate_fn
from Preprocessing.cache_audio import make_audio_sampler

CACHE_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
device = torch.device("cuda")

with open(r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave\train\index.json") as f:
    data = json.load(f)

attacks = sorted(set([d["attack"] for d in data]))
print(attacks)
model = AudioResNet18(pretrained=True).to(device)
model.train()

dataset = ASVSpoofDataset(CACHE_ROOT, split="train", augment=True, target_length=None)
sampler = make_audio_sampler([{"label": l} for l in dataset.labels])

loader = DataLoader(dataset, batch_size=16, sampler=sampler,
                    num_workers=0,   # 0 workers so errors surface cleanly
                    collate_fn=collate_fn)

for batch_idx, (x, y) in enumerate(loader):
    # check input
    print(f"batch {batch_idx:04d} | x shape={tuple(x.shape)} "
          f"min={x.min():.3f} max={x.max():.3f} "
          f"nan={torch.isnan(x).any().item()} "
          f"inf={torch.isinf(x).any().item()}")

    if torch.isnan(x).any() or torch.isinf(x).any():
        print("  ^^^ BAD INPUT BATCH")
        break

    x = x.to(device)
    with torch.no_grad():
        out = model(x)

    if torch.isnan(out).any():
        print(f"  ^^^ NaN OUTPUT for batch {batch_idx}")
        print(f"  x stats: min={x.min():.4f} max={x.max():.4f} std={x.std():.4f}")
        print(f"  x has inf: {torch.isinf(x).any().item()}")
        # check intermediate features
        feats = model.backbone(x)
        print(f"  backbone output: min={feats.min():.4f} max={feats.max():.4f} "
              f"nan={torch.isnan(feats).any().item()}")
        break

    if batch_idx >= 200:
        print("200 batches checked — no NaN found")
        break
