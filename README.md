# Multimodal Deepfake Detection — MTech Thesis

A three-stage pipeline for audio-visual deepfake detection combining independently trained video and audio models through score-level late fusion, evaluated on the FakeAVCeleb dataset.

> **Hardware:** All experiments conducted on NVIDIA GTX 1650 Ti (4 GB VRAM), Intel Core i5-10300H, 16 GB RAM.

---

## Repository Structure

```
├── model/
│   ├── video/                          # Video model — training, evaluation, preprocessing
│   │   ├── Preprocessing/              # Face extraction and caching pipeline
│   │   │   ├── video_preprocessing.py  # MTCNN face detection, frame extraction
│   │   │   ├── cached_FakeAVCelebs.py  # FakeAVCeleb cache builder
│   │   │   ├── FF++c23_pre.py          # FF++ preprocessing
│   │   │   ├── celeb_DB_pre.py         # CelebDF preprocessing
│   │   │   ├── face_cache.py           # Cache management
│   │   │   └── png_to_npy.py           # Frame format conversion
│   │   ├── video_model.py              # R3D-18 architecture definition
│   │   ├── train.py                    # Stage 1: training on FF++ + CelebDF
│   │   ├── finetune_video.py           # Stage 2: fine-tuning on FakeAVCeleb
│   │   ├── eval_video.py               # General evaluation script
│   │   ├── test_video_FF++C23.py       # In-distribution eval on FF++
│   │   ├── test_video_CelebDb.py       # In-distribution eval on CelebDF
│   │   ├── test_FakeAVCelebs.py        # Zero-shot eval on FakeAVCeleb
│   │   ├── test_finetuned_video.py     # Post-finetuning eval
│   │   ├── val_fakeav_slipt_video.py   # Val split inference (for fusion)
│   │   ├── eval_outputs/               # Saved scores and labels
│   │   │   ├── FF++C23/                # FF++ eval outputs
│   │   │   ├── CelebDB/                # CelebDF eval outputs
│   │   │   └── FakeAVCeleb/            # FakeAVCeleb eval outputs
│   │   ├── train.txt                   # Stage 1 training log
│   │   └── finetune_fakeav.txt         # Stage 2 fine-tuning log
│   │
│   ├── audio/                          # Audio model — training, evaluation, preprocessing
│   │   ├── Preprocessing/              # Log-mel spectrogram extraction
│   │   ├── audio_model.py              # AudioResNet18 architecture definition
│   │   ├── train_audio.py              # Stage 1: training on ASVspoof2019 + ITW
│   │   ├── data_loader.py              # Dataset loading and sampling
│   │   ├── test_asvspoof.py            # In-distribution eval on ASVspoof2019
│   │   ├── test_itw.py                 # In-distribution eval on ITW
│   │   ├── test_fakeavceleb.py         # Zero-shot eval on FakeAVCeleb
│   │   ├── test_audio_split.py         # Split-aligned subset eval
│   │   ├── val_fakeav_slipt_audio.py   # Val split inference (for fusion)
│   │   ├── eval_outputs/               # ASVspoof eval outputs
│   │   ├── eval_outputs_itw/           # ITW eval outputs
│   │   ├── eval_fakeavceleb_test_split/ # FakeAVCeleb test split outputs
│   │   ├── eval_val_for_alpha/         # Val split scores for alpha tuning
│   │   ├── output.txt                  # Audio training log
│   │   └── training_log_combined.csv   # Per-epoch training metrics
│   │
│   └── Fine tune/                      # FakeAVCeleb split management
│       ├── create_splits.py            # Speaker-level train/val/test split creation
│       ├── align_and_fuse_test.py      # Test split alignment helper
│       └── splits.json                 # Final split assignments
│
├── Fusion_Layer/                       # Stage 3: score alignment and fusion
│   ├── align_scores.py                 # file_id-based score alignment (val + test)
│   ├── fuse_weighted.py                # Weighted score fusion (alpha sweep on val)
│   ├── fuse_learned.py                 # LR and MLP fusion (trained on val, eval on test)
│   ├── plot_results.py                 # Thesis figures generation
│   ├── fusion_utility.py               # Shared metrics utilities (EER, threshold)
│   ├── fusion_master_index.py          # Index management for alignment
│   ├── fusion_slipt.py                 # Split handling for fusion pipeline
│   ├── splits.json                     # Dataset split file (shared with Fine tune/)
│   ├── aligned_alpha_val_outputs/      # Aligned val scores (input to fusion training)
│   │   ├── aligned_video_scores.npy
│   │   ├── aligned_audio_scores.npy
│   │   ├── aligned_labels.npy
│   │   └── aligned_meta.json
│   ├── aligned_alpha_test_outputs/     # Aligned test scores (input to fusion eval)
│   │   ├── aligned_video_scores.npy
│   │   ├── aligned_audio_scores.npy
│   │   ├── aligned_labels.npy
│   │   └── aligned_meta.json
│   └── results/
│       ├── weighted/                   # Weighted fusion outputs
│       │   ├── weighted_fusion_results.json
│       │   ├── weighted_best_test_scores.npy
│       │   └── weighted_auc_curve.png
│       └── learned/                    # LR and MLP fusion outputs
│           ├── learned_fusion_results.json
│           ├── lr_fusion_test_scores.npy
│           ├── mlp_fusion_test_scores.npy
│           ├── lr_model.pkl
│           ├── mlp_model.pkl
│           └── roc_comparison.png
│── fusion_results/
|    |── results/
|        |── plots/
|            |── fig1_overall_auc.png
|            |── fig2_per_attack_auc.png
|            |── fig3_alpha_curve.png
|            |── fig5_speaker_scatter.png
|
├── output.png                          # System output sample
└── system blueprint.png                # Pipeline architecture diagram
```

---

## Pipeline Overview

### Stage 1 — Unimodal Training (Cross-Dataset Baselines)

**Video model** trained on FF++ + CelebDF (8,843 samples, 2,000 resampled per epoch).
- Architecture: R3D-18, 24 frames × 224×224, pretrained on Kinetics-400
- Best val AUC on CelebDF: **0.9873** (epoch 14)
- Zero-shot AUC on FakeAVCeleb: **0.5352** (near random chance)

**Audio model** trained on ASVspoof2019-LA + In-The-Wild (41,476 samples).
- Architecture: AudioResNet18, 128-band log-mel spectrogram
- Best checkpoint: epoch 7, combined EER **0.1273**
- Zero-shot AUC on FakeAVCeleb: **0.7827**

### Stage 2 — Domain Adaptation

Video model fine-tuned on FakeAVCeleb training split (same architecture, LR = 1e-5).
- Best val AUC: **0.8474**
- Test AUC: **0.8507** (+0.315 from zero-shot baseline)

Audio model deliberately **not** fine-tuned — preserved as zero-shot cross-domain baseline.

### Stage 3 — Score-Level Fusion

Scores aligned by `file_id` string intersection (166 test orphans dropped).
Aligned test set: **6,294 samples** (3,005 real, 3,289 fake).

| Method | AUC | EER | Accuracy |
|---|---|---|---|
| Audio Only | 0.7827 | 0.2825 | 72.0% |
| Video Only | 0.8507 | 0.2445 | 75.6% |
| Weighted Fusion (α = 0.45) | 0.8837 | 0.2077 | 79.97% |
| LR Fusion (C = 2.0) | 0.8836 | 0.2029 | 79.84% |
| **MLP Fusion (64, 32)** | **0.8863** | **0.2016** | **80.04%** |

Exceeds VFD (2022) AUC of 0.8611 with simpler late fusion architecture.

---

## Reproducing the Experiments

### 1. Preprocessing

```bash
# Video — extract and cache face frames
python model/video/Preprocessing/video_preprocessing.py

# Audio — log-mel spectrograms are generated on-the-fly during training
```

### 2. Stage 1 Training

```bash
# Video model
python model/video/train.py

# Audio model
python model/audio/train_audio.py
```

### 3. Stage 2 Fine-tuning

```bash
python model/video/finetune_video.py
```

### 4. Inference (generate scores for fusion)

```bash
# Val split — needed for alpha tuning and fusion head training
python model/video/val_fakeav_slipt_video.py
python model/audio/val_fakeav_slipt_audio.py

# Test split — needed for final evaluation
python model/video/test_finetuned_video.py
python model/audio/test_fakeavceleb.py
```

### 5. Score Alignment

Run twice — once for val, once for test (edit the split flag inside the script):

```bash
python Fusion_Layer/align_scores.py
```

### 6. Fusion

```bash
# Weighted fusion (alpha sweep on val, report on test)
python Fusion_Layer/fuse_weighted.py

# Learned fusion (LR + MLP, trained on val, evaluated on test)
python Fusion_Layer/fuse_learned.py
```

### 7. Generate Figures

```bash
python Fusion_Layer/plot_results.py
```

---

## Key Design Decisions

**Score alignment by file_id.** Audio and video inference pipelines produce caches in different orders. Positional alignment would silently corrupt ~all pairs. Both pipelines save `file_id` strings alongside scores; alignment is computed as a set intersection.

**Strict val/test separation.** Alpha sweep, LR regularisation search, and MLP architecture search all use the val split only. The test split is consulted exactly once per method after all hyperparameters are locked.

**Audio not fine-tuned on FakeAVCeleb.** This is intentional — it preserves a clean cross-domain generalization baseline and demonstrates the audio-video generalization asymmetry (video: AUC 0.53 → 0.85 with fine-tuning; audio: AUC 0.78 zero-shot without fine-tuning).

---

## Dependencies
Note: This project uses CUDA for GPU-accelerated compute. 
An NVIDIA GPU is strongly recommended for replication — training times on CPU will be prohibitively long. I
f no NVIDIA GPU is available, PyTorch will automatically fall back to CPU.

```
torch                # Deep learning framework — video and audio model training and inference
torchvision          # R3D-18 backbone, image transforms, and dataset utilities
torchaudio           # Audio loading and resampling
numpy                # Array operations for score alignment and fusion
scikit-learn         # Logistic regression, MLP, cross-validation, and metrics (AUC, EER)
librosa              # Log-mel spectrogram extraction for audio preprocessing
facenet-pytorch      # MTCNN face detection for video frame extraction
opencv-python        # Frame reading, resizing, and face region cropping
matplotlib           # Plotting — ROC curves, alpha sweep, per-attack bar charts
tqdm                 # Progress bars during inference and training loops
```

---

## Dataset Access

| Dataset | Used for | Link |
|---|---|---|
| FaceForensics++ | Video Stage 1 training | [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23?resource=download) |
| CelebDF-v2 | Video Stage 1 training + validation | [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) |
| ASVspoof2019-LA | Audio Stage 1 training | [Official site](https://www.asvspoof.org/) |
| In-The-Wild | Audio Stage 1 training | [Kaggle](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset) |
| FakeAVCeleb | Evaluation benchmark | [Kaggle](https://www.kaggle.com/datasets/shreyaty08/fakeavceleb) |

---

## Citation

If you use this code or results, please cite this thesis:

```
Karandeep Singh
Multimodal Deepfake Detection via Score-Level Fusion of
Independently Trained Audio and Video Models
MTech Thesis, [Gandwana University,Maharashtra-India], 2026
```
