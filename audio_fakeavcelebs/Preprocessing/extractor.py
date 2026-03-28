# extractor.py
import numpy as np
import librosa


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
FIXED_LENGTH_SEC = 4
MAX_SAMPLES = SAMPLE_RATE * FIXED_LENGTH_SEC

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000

EPS = 1e-10
NORM_EPS = 1e-6


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def pad_or_truncate(audio: np.ndarray) -> np.ndarray:
    length = audio.shape[0]

    if length > MAX_SAMPLES:
        return audio[:MAX_SAMPLES]

    if length < MAX_SAMPLES:
        pad_width = MAX_SAMPLES - length
        return np.pad(audio, (0, pad_width))

    return audio


def compute_log_mel(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    # ✅ RANDOM CROP
    length = audio.shape[0]
    if length > MAX_SAMPLES:
        start = np.random.randint(0, length - MAX_SAMPLES)
        audio = audio[start:start + MAX_SAMPLES]
    elif length < MAX_SAMPLES:
        pad_width = MAX_SAMPLES - length
        audio = np.pad(audio, (0, pad_width))

    # ✅ NORMALIZE AUDIO (critical)
    audio = audio - np.mean(audio)
    audio = audio / (np.std(audio) + 1e-6)

    # Mel
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    mel = np.maximum(mel, EPS)
    log_mel = librosa.power_to_db(mel, ref=1.0)

    # ✅ Normalize spectrogram
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    log_mel = (log_mel - mean) / (std + NORM_EPS)

    # ✅ TIME MASKING (anti-leak)
    T = log_mel.shape[1]
    if T > 40:
        t = np.random.randint(10, 30)
        t0 = np.random.randint(0, T - t)
        log_mel[:, t0:t0+t] = 0

    return log_mel.astype(np.float32)


def config_signature() -> str:
    return (
        f"sr={SAMPLE_RATE}|"
        f"fft={N_FFT}|"
        f"hop={HOP_LENGTH}|"
        f"mels={N_MELS}|"
        f"fmin={FMIN}|"
        f"fmax={FMAX}|"
        f"len={FIXED_LENGTH_SEC}"
    )
