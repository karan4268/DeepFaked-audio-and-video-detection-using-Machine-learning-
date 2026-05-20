# =============================================================================
# align_scores.py
#
# Aligns audio and video scores by matching on file_id
# (e.g. "RealVideo-RealAudio/African/men/id00476/00109.mp4")
#
# Both audio and video caches share the same index.json with the same
# file_id field — this is the correct alignment key since the `file`
# hash differs between caches (different content = different md5).
#
# Inputs:
#   - Video raw scores/labels/file_ids  (from val_fakeav_slipt_video.py)
#   - Audio raw scores/labels/file_ids  (from val_fakeav_slipt_audio.py)
#   - Shared index.json                 (video cache index — has all samples)
#
# Outputs (in OUTPUT_DIR):
#   - aligned_video_scores.npy
#   - aligned_audio_scores.npy
#   - aligned_labels.npy
#   - aligned_meta.json   (file_id, attack, speaker per sample)
#=============================================================================

#-------------------------------------------------------------------------------------------------------------
# =============================================================================
# NOTE:- RUN FOR VAL FIRST AND THEN CHANGE TO TEST FILES AND RUN AGAIN. 
# BECAUSE FUSION WEIGHTED AND LEARNED FILES NEED BOTH VAL AND TEST ALIGNED FILES TO WORK PROPERLY.
# =============================================================================
#-------------------------------------------------------------------------------------------------------------

import os
import json
import numpy as np

# =============================================================================
# PATHS — edit these to match your setup
# =============================================================================

VIDEO_SCORES_NPY   = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\eval_Fakeav_test_split\after_finetuning\finetuned_video_scores.npy"
)
VIDEO_LABELS_NPY   = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\eval_Fakeav_test_split\after_finetuning\finetuned_video_labels.npy"
)
# FIX: this file contains file_id strings (not integer indices) — loaded correctly below
VIDEO_FILEIDS_NPY  = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\video\eval_Fakeav_test_split\after_finetuning\finetuned_video_file_ids.npy"
)

AUDIO_SCORES_NPY   = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\eval_fakeavceleb_test_split\audio_test_scores.npy"
)
AUDIO_LABELS_NPY   = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\eval_fakeavceleb_test_split\audio_test_labels.npy"
)
# FIX: use the saved audio file_ids instead of assuming index order
AUDIO_FILEIDS_NPY  = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Models\audio\eval_fakeavceleb_test_split\audio_test_file_ids.npy"
)

# Both caches share the same similar index.json structure with the same file_id field,
# but audio side has 3 less samples. Using video cache's index.json since it has all samples.
SHARED_INDEX_JSON  = r"D:\FakeAVCache\Video\video_index.json"

OUTPUT_DIR = (
    r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer"
    r"\Fusion_Layer\aligned_alpha_test_outputs"
)

# =============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load shared index  (used only for meta lookup — not for ordering)
    # -------------------------------------------------------------------------
    print("[INFO] Loading shared index.json ...")
    with open(SHARED_INDEX_JSON, "r") as f:
        index = json.load(f)

    print(f"  Total entries in index : {len(index)}")
    print(f"  Sample file_id         : {index[0]['file_id']}")

    # Build a file_id → meta dict for fast lookup
    index_by_fileid = {entry["file_id"]: entry for entry in index}

    # -------------------------------------------------------------------------
    # Load video outputs
    # FIX: VIDEO_FILEIDS_NPY contains file_id strings saved by val_fakeav_slipt_video.py
    #      — load with allow_pickle=True, do NOT call .astype(int)
    # -------------------------------------------------------------------------
    print("\n[INFO] Loading video scores/labels/file_ids ...")
    vid_scores   = np.load(VIDEO_SCORES_NPY)
    vid_labels   = np.load(VIDEO_LABELS_NPY)
    vid_file_ids = np.load(VIDEO_FILEIDS_NPY, allow_pickle=True)  # string array
    print(f"  Video samples : {len(vid_scores)}")

    vid_fileid_to_score = {}
    vid_fileid_to_label = {}
    vid_fileid_to_meta  = {}

    for pos, file_id in enumerate(vid_file_ids):
        vid_fileid_to_score[file_id] = float(vid_scores[pos])
        vid_fileid_to_label[file_id] = int(vid_labels[pos])
        vid_fileid_to_meta[file_id]  = index_by_fileid.get(
            file_id, {"file_id": file_id}
        )

    print(f"  Unique video file_ids  : {len(vid_fileid_to_score)}")

    # -------------------------------------------------------------------------
    # Load audio outputs
    # FIX: use AUDIO_FILEIDS_NPY saved by val_fakeav_slipt_audio.py instead of
    #      assuming audio scores are in the same order as index.json (they are not —
    #      audio had 6336 samples vs video's 6335, so positional mapping was wrong)
    # -------------------------------------------------------------------------
    print("\n[INFO] Loading audio scores/labels/file_ids ...")
    aud_scores   = np.load(AUDIO_SCORES_NPY)
    aud_labels   = np.load(AUDIO_LABELS_NPY)
    aud_file_ids = np.load(AUDIO_FILEIDS_NPY, allow_pickle=True)  # string array
    print(f"  Audio samples : {len(aud_scores)}")

    if len(aud_scores) != len(aud_file_ids):
        print(f"  [WARN] audio scores length ({len(aud_scores)}) != "
              f"audio file_ids length ({len(aud_file_ids)}). "
              f"Using min of the two.")

    aud_fileid_to_score = {}
    aud_fileid_to_label = {}

    n_aud = min(len(aud_scores), len(aud_file_ids))
    for pos in range(n_aud):
        file_id = aud_file_ids[pos]
        aud_fileid_to_score[file_id] = float(aud_scores[pos])
        aud_fileid_to_label[file_id] = int(aud_labels[pos])

    print(f"  Unique audio file_ids  : {len(aud_fileid_to_score)}")

    # -------------------------------------------------------------------------
    # Find intersection
    # -------------------------------------------------------------------------
    common_ids = sorted(
        set(vid_fileid_to_score.keys()) & set(aud_fileid_to_score.keys())
    )
    print(f"\n[INFO] Common samples (intersection) : {len(common_ids)}")
    print(f"  Video-only             : {len(vid_fileid_to_score) - len(common_ids)}")
    print(f"  Audio-only             : {len(aud_fileid_to_score) - len(common_ids)}")

    if len(common_ids) == 0:
        vid_sample = list(vid_fileid_to_score.keys())[:3]
        aud_sample = list(aud_fileid_to_score.keys())[:3]
        print("\n[ERROR] Zero intersection — sample file_ids from each side:")
        print(f"  Video : {vid_sample}")
        print(f"  Audio : {aud_sample}")
        print("\nThe file_id fields don't match. Check SHARED_INDEX_JSON is correct.")
        return

    # -------------------------------------------------------------------------
    # Build aligned arrays
    # -------------------------------------------------------------------------
    aligned_vid_scores = []
    aligned_aud_scores = []
    aligned_labels     = []
    aligned_meta       = []

    label_mismatches = 0

    for fid in common_ids:
        v_score = vid_fileid_to_score[fid]
        a_score = aud_fileid_to_score[fid]
        v_label = vid_fileid_to_label[fid]
        a_label = aud_fileid_to_label[fid]

        if v_label != a_label:
            label_mismatches += 1

        aligned_vid_scores.append(v_score)
        aligned_aud_scores.append(a_score)
        aligned_labels.append(v_label)   # use video label as source of truth
        aligned_meta.append(vid_fileid_to_meta[fid])

    if label_mismatches > 0:
        print(f"[WARN] {label_mismatches} label mismatches — using video labels.")

    aligned_vid_scores = np.array(aligned_vid_scores, dtype=np.float32)
    aligned_aud_scores = np.array(aligned_aud_scores, dtype=np.float32)
    aligned_labels     = np.array(aligned_labels,     dtype=np.int32)

    n_real = (aligned_labels == 0).sum()
    n_fake = (aligned_labels == 1).sum()
    print(f"\n[INFO] Aligned dataset — real: {n_real}  fake: {n_fake}  total: {len(aligned_labels)}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, "aligned_video_scores.npy"), aligned_vid_scores)
    np.save(os.path.join(OUTPUT_DIR, "aligned_audio_scores.npy"), aligned_aud_scores)
    np.save(os.path.join(OUTPUT_DIR, "aligned_labels.npy"),       aligned_labels)

    with open(os.path.join(OUTPUT_DIR, "aligned_meta.json"), "w") as f:
        json.dump(aligned_meta, f, indent=2)

    print(f"\n[INFO] Saved aligned outputs → {OUTPUT_DIR}")
    print("  aligned_video_scores.npy")
    print("  aligned_audio_scores.npy")
    print("  aligned_labels.npy")
    print("  aligned_meta.json")


if __name__ == "__main__":
    main()