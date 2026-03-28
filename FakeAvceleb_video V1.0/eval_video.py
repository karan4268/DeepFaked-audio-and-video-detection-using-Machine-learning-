# eval_video.py
# Evaluate a video model on any DataLoader (binary or multi-class)
# Metrics: Accuracy, Precision, Recall, F1-score, AUC

#| Metric    | Purpose                       |
#| --------- | ----------------------------- |
#| ROC-AUC   | primary detection performance |
#| Accuracy  | overall correctness           |
#| F1-score  | balance of precision/recall   |
#| Precision | fake prediction reliability   |
#| Recall    | fake detection ability        |


import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np


def evaluate(model, data_loader, device):
    model.eval()

    y_true = []
    y_score = []
    y_pred = []

    with torch.no_grad():
        for frames, labels in data_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(frames)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)

    # ------------------------------------------------
    # Accuracy
    # ------------------------------------------------
    acc = accuracy_score(y_true, y_pred)

    # ------------------------------------------------
    # Precision, Recall, F1-score
    # ------------------------------------------------
    if len(np.unique(y_true)) > 1:
        precision = precision_score(y_true, y_pred, average='binary' if y_score.shape[1]==2 else 'macro')
        recall = recall_score(y_true, y_pred, average='binary' if y_score.shape[1]==2 else 'macro')
        f1 = f1_score(y_true, y_pred, average='binary' if y_score.shape[1]==2 else 'macro')
    else:
        print("[WARNING] Only one class present in evaluation. Precision/Recall/F1 undefined.")
        precision, recall, f1 = 0.0, 0.0, 0.0

    # ------------------------------------------------
    # ROC AUC
    # ------------------------------------------------
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print("[WARNING] Only one class present in evaluation. AUC undefined.")
        auc = 0.5
    elif y_score.shape[1] == 2:
        auc = roc_auc_score(y_true, y_score[:, 1])
    else:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }