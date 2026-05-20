# eval_video.py
# Evaluate a video model on any DataLoader
#
# Updated:
# - AMP support
# - Test-Time Augmentation (temporal jitter)
# - Optional return of raw scores (for threshold optimization)
# - Robust AUC handling
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(
    model,
    data_loader,
    device,
    tta=1,                     # number of temporal augmentations
    threshold=0.5,             # fixed threshold (do NOT optimize here)
    return_scores=False        # for post-training threshold tuning
):
    """
    Evaluates a video model.

    Args:
        model (nn.Module)
        data_loader (DataLoader)
        device (torch.device)
        tta (int): number of temporal augmentations
        threshold (float): classification threshold
        return_scores (bool): return raw outputs for analysis

    Returns:
        acc (float)
        auc (float)
        (optional) y_true, y_score
    """

    model.eval()
    y_true, y_score, y_pred = [], [], []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for frames, labels in data_loader:

            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            probs_list = []

            # --------------------------------------------------
            # TTA loop (temporal jitter)
            # --------------------------------------------------
            for _ in range(tta):

                if tta > 1:
                    T = frames.shape[2]

                    jitter = torch.randint(
                        -2, 3, (T,),
                        device=frames.device
                    )

                    idx = torch.clamp(
                        torch.arange(T, device=frames.device) + jitter,
                        0, T - 1
                    )

                    frames_aug = frames[:, :, idx, :, :]
                else:
                    frames_aug = frames

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(frames_aug)
                    probs = F.softmax(logits, dim=1)[:, 1]

                probs_list.append(probs)

            # average over TTA
            probs_mean = torch.stack(probs_list, dim=0).mean(dim=0)

            preds = (probs_mean > threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs_mean.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0

    if return_scores:
        return acc, auc, np.array(y_true), np.array(y_score)

    return acc, auc