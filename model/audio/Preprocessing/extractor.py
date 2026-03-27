# extractor.py
# =============================================================================
# Dynamic Feature Extractor (FIXED + STABLE + ENERGY PRESERVING)
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
FMAX       = 7600   # slightly below Nyquist for stability

EPS = 1e-6


# ===========================================================
# AUDIO SAFETY
# ===========================================================

def safe_audio(audio: np.ndarray) -> np.ndarray:
    """
    Preserve amplitude while removing invalid values.
    """
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
# FEATURE COMPUTE
# ===========================================================

def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram with stable scaling.
    Preserves amplitude information (no per-sample z-norm).
    """

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

    # Stable log scale
    log_mel = librosa.power_to_db(mel, ref=1.0)

    # FIX: preserve energy instead of z-normalizing
    # Typical range ~[-80, 0] → scale to [-1, 0]
    log_mel = log_mel / 80.0

    # Clip for numerical stability
    log_mel = np.clip(log_mel, -1.0, 0.0)

    # Final safety
    log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

    return log_mel.astype(np.float32)


# ===========================================================
# ALIAS
# ===========================================================

def compute_log_mel_from_audio(audio: np.ndarray) -> np.ndarray:
    return compute_log_mel(audio)


# ===========================================================
# SIGNATURE
# ===========================================================

def config_signature():
    return "waveform_cache_v2_energy_preserved"
