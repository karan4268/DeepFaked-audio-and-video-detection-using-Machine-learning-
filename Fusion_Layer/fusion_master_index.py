import os
import json

# =========================
# PATHS
# =========================

SPLITS_JSON = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Fusion_Layer\splits.json"

# Updated to match fixed cache script output filenames
VIDEO_INDEX = r"D:\FakeAVCache\Video\video_index.json"
AUDIO_INDEX = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave\fakeav_test\audio_index.json"

OUTPUT_PATH = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Fusion_Layer\fusion_master_index.json"


# =========================
# LOAD HELPERS
# =========================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# MAIN
# =========================

def main():

    print("[INFO] Loading indices...")

    splits      = load_json(SPLITS_JSON)
    video_index = load_json(VIDEO_INDEX)
    audio_index = load_json(AUDIO_INDEX)

    # -------------------------------------------------
    # Split handling (source of truth = splits.json)
    # -------------------------------------------------

    test_indices = set(splits["test_indices"])

    test_file_ids = set(
        video_index[i]["file_id"]
        for i in test_indices
        if i < len(video_index)
    )

    # -------------------------------------------------
    # Build lookup maps (file_id → metadata)
    # -------------------------------------------------

    video_map = {x["file_id"]: x for x in video_index}
    audio_map = {x["file_id"]: x for x in audio_index}

    # -------------------------------------------------
    # Master union space (NO INTERSECTION EVER)
    # -------------------------------------------------

    all_file_ids = (
        set(video_map.keys())
        | set(audio_map.keys())
        | test_file_ids
    )

    print(f"[INFO] Total unique IDs: {len(all_file_ids)}")

    # -------------------------------------------------
    # Build fusion master index
    # -------------------------------------------------

    fusion_index = []

    for fid in all_file_ids:

        v = video_map.get(fid)
        a = audio_map.get(fid)

        base = v or a  # fallback source

        entry = {
            "file_id":     fid,

            # label priority: video → audio → None
            "label":       base["label"] if base else None,

            # modality availability flags
            "video_valid": v is not None,
            "audio_valid": a is not None,

            # resolvable load paths (None if modality missing)
            # audio: np.load(os.path.join(AUDIO_CACHE_ROOT, entry["audio_file"]))
            # video: np.load(os.path.join(VIDEO_CACHE_ROOT, entry["video_file"]))
            "audio_file":  a["file"] if a else None,   # e.g. "fakeav_test/real/abc123.npy"
            "video_file":  v["file"] if v else None,   # e.g. "abc123/frames.npy"

            # metadata (safe fallback)
            "speaker":     base.get("speaker") if base else None,
            "attack":      base.get("attack")  if base else None,

            # split membership (ground truth test set)
            "is_test":     fid in test_file_ids,
        }

        fusion_index.append(entry)

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(fusion_index, f, indent=2)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    paired      = sum(1 for x in fusion_index if x["video_valid"] and x["audio_valid"])
    video_only  = sum(1 for x in fusion_index if x["video_valid"] and not x["audio_valid"])
    audio_only  = sum(1 for x in fusion_index if x["audio_valid"] and not x["video_valid"])
    test_count  = sum(1 for x in fusion_index if x["is_test"])
    real_count  = sum(1 for x in fusion_index if x["label"] == 0)
    fake_count  = sum(1 for x in fusion_index if x["label"] == 1)

    print("\n[DONE] Fusion master index created")
    print(f"[INFO] Saved to: {OUTPUT_PATH}")
    print(f"[INFO] Total entries : {len(fusion_index)}")
    print("\n[STATS]")
    print(f"  Both modalities : {paired}")
    print(f"  Video-only      : {video_only}")
    print(f"  Audio-only      : {audio_only}")
    print(f"  Test set        : {test_count}")
    print(f"  Real            : {real_count}")
    print(f"  Fake            : {fake_count}")

    # Sanity check: warn if any paired entry has a label mismatch
    mismatches = [
        fid for fid, v, a in (
            (fid, video_map.get(fid), audio_map.get(fid))
            for fid in set(video_map) & set(audio_map)
        )
        if v["label"] != a["label"]
    ]
    if mismatches:
        print(f"\n[WARNING] Label mismatches across modalities: {len(mismatches)}")
        for fid in mismatches[:5]:
            print(f"  file_id={fid}  video_label={video_map[fid]['label']}  audio_label={audio_map[fid]['label']}")
    else:
        print("\n[OK] No label mismatches across paired entries")


if __name__ == "__main__":
    main()