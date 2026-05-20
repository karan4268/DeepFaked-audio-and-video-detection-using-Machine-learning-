# =============================================================================
# verify_cache_alignment.py
# Run this AFTER both caches are built to confirm they are fusion-ready.
# =============================================================================

import os
import json
import numpy as np
from collections import defaultdict

# =============================================================================
# PATHS  — edit if yours differ
# =============================================================================

AUDIO_CACHE_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Audio\cache_wave"
VIDEO_CACHE_ROOT = r"D:\FakeAVCache\Video"

AUDIO_INDEX = os.path.join(AUDIO_CACHE_ROOT, "fakeav_test", "audio_index.json")
VIDEO_INDEX = os.path.join(VIDEO_CACHE_ROOT, "video_index.json")

# How many paired entries to spot-check by actually loading the .npy files
NPY_SPOT_CHECK = 20

# =============================================================================
# HELPERS
# =============================================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Index files exist
    # ------------------------------------------------------------------
    section("1. INDEX FILE EXISTS")

    for label, path in [("Audio index", AUDIO_INDEX), ("Video index", VIDEO_INDEX)]:
        exists = os.path.isfile(path)
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {label}: {path}")

    if not os.path.isfile(AUDIO_INDEX) or not os.path.isfile(VIDEO_INDEX):
        print("\n[ABORT] Fix missing index files before continuing.")
        return

    audio_entries = load_json(AUDIO_INDEX)
    video_entries = load_json(VIDEO_INDEX)

    # ------------------------------------------------------------------
    # 2. Schema check — required fields in every entry
    # ------------------------------------------------------------------
    section("2. SCHEMA CHECK")

    AUDIO_REQUIRED = {"file_id", "file", "label", "speaker", "attack", "dataset", "modality"}
    VIDEO_REQUIRED = {"file_id", "file", "label", "speaker", "attack", "dataset", "modality"}

    def check_schema(entries, required, name):
        bad = []
        for i, e in enumerate(entries):
            missing = required - set(e.keys())
            if missing:
                bad.append((i, missing))
        if bad:
            print(f"  ❌ {name}: {len(bad)} entries missing fields")
            for i, m in bad[:5]:
                print(f"     entry[{i}] missing: {m}")
        else:
            print(f"  ✅ {name}: all {len(entries)} entries have required fields")

    check_schema(audio_entries, AUDIO_REQUIRED, "Audio index")
    check_schema(video_entries, VIDEO_REQUIRED, "Video index")

    # ------------------------------------------------------------------
    # 3. Modality field correctness
    # ------------------------------------------------------------------
    section("3. MODALITY FIELD VALUES")

    wrong_audio = [e for e in audio_entries if e.get("modality") != "audio"]
    wrong_video = [e for e in video_entries if e.get("modality") != "video"]

    print(f"  {'✅' if not wrong_audio else '❌'} Audio entries with modality != 'audio' : {len(wrong_audio)}")
    print(f"  {'✅' if not wrong_video else '❌'} Video entries with modality != 'video' : {len(wrong_video)}")

    # ------------------------------------------------------------------
    # 4. file_id format consistency
    # ------------------------------------------------------------------
    section("4. FILE_ID FORMAT CONSISTENCY")

    audio_ids = {e["file_id"] for e in audio_entries}
    video_ids = {e["file_id"] for e in video_entries}

    # Sample a few to confirm they look like relative paths
    print("  Audio file_id samples:")
    for e in audio_entries[:3]:
        print(f"    {e['file_id']}")
    print("  Video file_id samples:")
    for e in video_entries[:3]:
        print(f"    {e['file_id']}")

    # ------------------------------------------------------------------
    # 5. Intersection / union coverage
    # ------------------------------------------------------------------
    section("5. COVERAGE")

    paired     = audio_ids & video_ids
    audio_only = audio_ids - video_ids
    video_only = video_ids - audio_ids

    print(f"  Audio entries     : {len(audio_ids)}")
    print(f"  Video entries     : {len(video_ids)}")
    print(f"  Paired (both)     : {len(paired)}")
    print(f"  Audio-only        : {len(audio_only)}")
    print(f"  Video-only        : {len(video_only)}")

    if audio_only:
        print(f"\n  [WARN] {len(audio_only)} file_ids have audio but no video cache")
        for fid in list(audio_only)[:3]:
            print(f"    {fid}")

    if video_only:
        print(f"\n  [WARN] {len(video_only)} file_ids have video but no audio cache")
        for fid in list(video_only)[:3]:
            print(f"    {fid}")

    # ------------------------------------------------------------------
    # 6. Label consistency across paired entries
    # ------------------------------------------------------------------
    section("6. LABEL CONSISTENCY (paired entries)")

    audio_map = {e["file_id"]: e for e in audio_entries}
    video_map = {e["file_id"]: e for e in video_entries}

    label_mismatches = []
    for fid in paired:
        al = audio_map[fid]["label"]
        vl = video_map[fid]["label"]
        if al != vl:
            label_mismatches.append((fid, al, vl))

    if label_mismatches:
        print(f"  ❌ Label mismatches: {len(label_mismatches)}")
        for fid, al, vl in label_mismatches[:5]:
            print(f"    file_id={fid}  audio_label={al}  video_label={vl}")
    else:
        print(f"  ✅ All {len(paired)} paired entries have matching labels")

    # ------------------------------------------------------------------
    # 7. Speaker consistency across paired entries
    # ------------------------------------------------------------------
    section("7. SPEAKER CONSISTENCY (paired entries)")

    speaker_mismatches = []
    for fid in paired:
        asp = audio_map[fid].get("speaker")
        vsp = video_map[fid].get("speaker")
        if asp != vsp:
            speaker_mismatches.append((fid, asp, vsp))

    if speaker_mismatches:
        print(f"  ❌ Speaker mismatches: {len(speaker_mismatches)}")
        for fid, a, v in speaker_mismatches[:5]:
            print(f"    file_id={fid}  audio={a}  video={v}")
    else:
        print(f"  ✅ All {len(paired)} paired entries have matching speakers")

    # ------------------------------------------------------------------
    # 8. attack / bonafide consistency across paired entries
    # ------------------------------------------------------------------
    section("8. ATTACK FIELD CONSISTENCY (paired entries)")

    attack_mismatches = []
    for fid in paired:
        aa = audio_map[fid].get("attack")
        va = video_map[fid].get("attack")
        if aa != va:
            attack_mismatches.append((fid, aa, va))

    if attack_mismatches:
        print(f"  ❌ Attack mismatches: {len(attack_mismatches)}")
        for fid, a, v in attack_mismatches[:5]:
            print(f"    file_id={fid}  audio={a}  video={v}")
    else:
        print(f"  ✅ All {len(paired)} paired entries have matching attack fields")

    # ------------------------------------------------------------------
    # 9. file path format check
    #    audio: "fakeav_test/{real|fake}/{hash}.npy"
    #    video: "{hash}/frames.npy"
    # ------------------------------------------------------------------
    section("9. FILE PATH FORMAT")

    audio_bad_fmt = [e for e in audio_entries if not e["file"].endswith(".npy")]
    video_bad_fmt = [e for e in video_entries if not e["file"].endswith("frames.npy")]

    print(f"  {'✅' if not audio_bad_fmt else '❌'} Audio paths ending in .npy       : {len(audio_entries) - len(audio_bad_fmt)}/{len(audio_entries)}")
    print(f"  {'✅' if not video_bad_fmt else '❌'} Video paths ending in frames.npy : {len(video_entries) - len(video_bad_fmt)}/{len(video_entries)}")

    if audio_bad_fmt:
        for e in audio_bad_fmt[:3]:
            print(f"    BAD audio file: {e['file']}")
    if video_bad_fmt:
        for e in video_bad_fmt[:3]:
            print(f"    BAD video file: {e['file']}")

    # ------------------------------------------------------------------
    # 10. Spot-check: actually load .npy files for N paired entries
    # ------------------------------------------------------------------
    section(f"10. NPY SPOT CHECK ({NPY_SPOT_CHECK} paired entries)")

    sample_ids = list(paired)[:NPY_SPOT_CHECK]
    audio_load_ok = 0
    video_load_ok = 0
    audio_shape_ok = 0
    video_shape_ok = 0
    load_errors = []

    for fid in sample_ids:
        a_entry = audio_map[fid]
        v_entry = video_map[fid]

        a_path = os.path.join(AUDIO_CACHE_ROOT, a_entry["file"])
        v_path = os.path.join(VIDEO_CACHE_ROOT, v_entry["file"])

        # Audio
        try:
            a_arr = np.load(a_path)
            audio_load_ok += 1
            # Expected: 1D int16 array, length between 16000 and 96000
            if a_arr.ndim == 1 and a_arr.dtype == np.int16 and 16000 <= len(a_arr) <= 96000:
                audio_shape_ok += 1
            else:
                load_errors.append(f"Audio shape/dtype wrong: {a_arr.shape} {a_arr.dtype} — {fid}")
        except Exception as e:
            load_errors.append(f"Audio load failed: {e} — {fid}")

        # Video
        try:
            v_arr = np.load(v_path)
            video_load_ok += 1
            # Expected: (24, 224, 224, 3) uint8
            if v_arr.shape == (24, 224, 224, 3) and v_arr.dtype == np.uint8:
                video_shape_ok += 1
            else:
                load_errors.append(f"Video shape/dtype wrong: {v_arr.shape} {v_arr.dtype} — {fid}")
        except Exception as e:
            load_errors.append(f"Video load failed: {e} — {fid}")

    print(f"  Audio — loaded: {audio_load_ok}/{NPY_SPOT_CHECK}  shape+dtype OK: {audio_shape_ok}/{NPY_SPOT_CHECK}")
    print(f"  Video — loaded: {video_load_ok}/{NPY_SPOT_CHECK}  shape+dtype OK: {video_shape_ok}/{NPY_SPOT_CHECK}")

    if load_errors:
        print(f"\n  ❌ Errors:")
        for err in load_errors:
            print(f"    {err}")
    else:
        print(f"\n  ✅ All spot-checked files loaded with correct shape and dtype")

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------
    section("FINAL VERDICT")

    all_ok = (
        not wrong_audio and
        not wrong_video and
        not label_mismatches and
        not speaker_mismatches and
        not attack_mismatches and
        not audio_bad_fmt and
        not video_bad_fmt and
        audio_load_ok == NPY_SPOT_CHECK and
        video_load_ok == NPY_SPOT_CHECK and
        audio_shape_ok == NPY_SPOT_CHECK and
        video_shape_ok == NPY_SPOT_CHECK
    )

    if all_ok:
        print("  ✅ CACHES ARE ALIGNED AND FUSION-READY")
    else:
        print("  ❌ ISSUES FOUND — review sections above before proceeding to fusion")

    print()


# =============================================================================

if __name__ == "__main__":
    main()