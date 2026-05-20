# Fusion Pipeline — README
=============================================

## Execution Order

Run scripts in this exact order:

```
1. align_scores.py
2. fuse_weighted.py
3. fuse_learned.py
4. plot_results.py
```

---

## Step 1 — align_scores.py

Matches audio and video scores by file hash (md5).
Audio has 21,563 samples, video has 21,566 — this finds the ~21,563 common ones.

**Edit these paths before running:**
- `VIDEO_SCORES_NPY`  → raw_scores.npy  from video eval
- `VIDEO_LABELS_NPY`  → raw_labels.npy  from video eval
- `VIDEO_INDICES_NPY` → raw_indices.npy from video eval
- `VIDEO_INDEX_JSON`  → D:\FakeAVCache\Video\index.json
- `AUDIO_SCORES_NPY`  → your audio model's saved scores .npy
- `AUDIO_LABELS_NPY`  → your audio model's saved labels .npy
- `AUDIO_INDEX_JSON`  → D:\FakeAVCache\Audio\index.json  (needs same file/hash field)

**Outputs → Models\fusion\aligned\**
```
aligned_video_scores.npy
aligned_audio_scores.npy
aligned_labels.npy
aligned_meta.json          ← file_id, attack, speaker per sample
```

---

## Step 2 — fuse_weighted.py

Sweeps alpha from 0.0 to 1.0 in 0.05 steps.
`fusion = alpha * audio + (1-alpha) * video`
Finds best alpha by AUC, prints full metrics, plots the curve.

**Outputs → Models\fusion\results\weighted\**
```
weighted_fusion_results.json
weighted_auc_curve.png
weighted_best_scores.npy
```

---

## Step 3 — fuse_learned.py

Trains Logistic Regression and MLP on [audio_score, video_score] features.
Uses 5-fold stratified cross-validation — OOF scores are unbiased.
Also trains final models on all data and saves them for inference.

**Outputs → Models\fusion\results\learned\**
```
learned_fusion_results.json
lr_fusion_scores.npy        ← OOF scores, use these for evaluation
mlp_fusion_scores.npy       ← OOF scores
lr_model.pkl                ← final model for inference
mlp_model.pkl               ← final model for inference
roc_comparison.png
```

---

## Step 4 — plot_results.py

Generates all thesis figures from the saved results.
Reads from all three output directories above.

**Outputs → Models\fusion\results\plots\**
```
fig1_overall_auc.png        ← bar chart: audio / video / LR / MLP
fig2_per_attack_auc.png     ← grouped bars per attack type
fig3_alpha_curve.png        ← AUC vs alpha sweep
fig4_roc_curves.png         ← ROC curves all methods
fig5_speaker_scatter.png    ← per-speaker audio AUC vs video AUC
```

---

## Audio Index JSON Note

`align_scores.py` expects your audio cache to have an `index.json` with the
same `file` (hash) field as the video cache. If your audio eval script saves
scores in a different order, check `AUDIO_INDEX_JSON` matches the order of
`AUDIO_SCORES_NPY`.

If your audio pipeline doesn't have an index.json with file hashes,
the simplest fix is to add the file hash to your audio eval output —
or align on `file_id` if both indexes share that field.

---

## Dependencies

```
pip install numpy scikit-learn matplotlib
```
All standard — no additional installs needed beyond what you already have.
