# extractor.py  (Preprocessing/extractor.py)
# =============================================================================
# Audio Feature Extractor — stable, energy-preserving log-mel
#
# CHANGE vs previous version:
#   [FIX 1] compute_log_mel() now guards against empty or near-silent audio.
#           An all-zeros array (e.g. from a corrupt cached file) produced a
#           mel spectrogram of all-EPS values, which after power_to_db and
#           /80.0 scaling produced a valid-looking -1.0 constant tensor.
#           The model then received an uninformative but loss-legal input,
#           wasting the batch step without crashing. Now such inputs are
#           detected early and a warning is printed so you can identify
#           and remove corrupt cache files.
#
# Everything else is unchanged — the energy-preserving normalization
# (log_mel / 80.0, clipped to [-1, 0]) is correct and should not be
# modified without rebuilding the cache.
# =============================================================================

import numpy as np
import librosa


# ===========================================================
# CONFIG
# ===========================================================

SAMPLE_RATE = 16000

N_FFT      = 1024
HOP_LENGTH = 256
N_MELS     = 128
FMIN       = 20
FMAX       = 7600   # slightly below Nyquist (8000 Hz) for numerical stability

EPS = 1e-6


# ===========================================================
# AUDIO SAFETY
# ===========================================================

def safe_audio(audio: np.ndarray) -> np.ndarray:
    """Remove invalid values while preserving amplitude."""
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


# ===========================================================
# LOAD AUDIO
# ===========================================================

def load_audio(wav_path: str) -> np.ndarray:
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        return safe_audio(audio)
    except Exception:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)


# ===========================================================
# FEATURE EXTRACTION
# ===========================================================

def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram with energy-preserving normalization.

    Output range: [-1.0, 0.0]
        -1.0 = silence / floor  (~-80 dB)
         0.0 = full scale        (0 dB, ref=1.0)

    Shape: (N_MELS, T) = (128, T)
    """

    # [FIX 1] Guard against empty or near-silent audio.
    # A zero array produces a valid mel but the model gets a useless
    # constant input with no diagnostic error. Warn instead.
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-7:
        print("[WARNING] compute_log_mel received near-silent audio — "
              "returning silence spectrogram. Check cache for corrupt files.")
        return np.full((N_MELS, 1), -1.0, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel = np.maximum(mel, EPS)

    # Convert to log scale (dB).  ref=1.0 anchors 0 dB at full scale.
    log_mel = librosa.power_to_db(mel, ref=1.0)

    # Scale to [-1, 0].  Typical speech range is ~[-80, 0] dB.
    # This preserves relative energy differences between samples,
    # unlike per-sample z-normalisation which would destroy them.
    log_mel = log_mel / 80.0
    log_mel = np.clip(log_mel, -1.0, 0.0)

    # Final NaN safety (should never trigger after the EPS guard above)
    log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

    return log_mel.astype(np.float32)


# ===========================================================
# ALIAS  (kept for backwards compatibility)
# ===========================================================

def compute_log_mel_from_audio(audio: np.ndarray) -> np.ndarray:
    return compute_log_mel(audio)


# ===========================================================
# CACHE SIGNATURE
# Changing this value invalidates all existing cached .npy files
# and forces a full rebuild on the next run of cache_audio.py.
# ===========================================================

def config_signature() -> str:
    return "waveform_cache_v2_energy_preserved"