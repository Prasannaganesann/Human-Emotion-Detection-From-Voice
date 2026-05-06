"""
feature_extraction.py
======================
Extracts a rich, multi-dimensional feature vector from preprocessed audio.

Features extracted per clip:
  • MFCC (40 coefficients) + Δ + ΔΔ
  • Chroma STFT (12 bins)
  • Mel Spectrogram (128 bins → statistics)
  • Spectral Contrast (7 bands)
  • Zero-Crossing Rate
  • RMS Energy
  • Spectral Rolloff
  • Tonnetz (6 components)

Each feature group is summarized as [mean, std, min, max] across time frames,
yielding a single flat vector per audio clip.
"""

import numpy as np
import librosa
from pathlib import Path
import logging
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS, N_MFCC,
    FMAX, FEATURE_CONFIG, FEATURE_NAMES_FILE
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Stat Aggregation Helper
# ─────────────────────────────────────────────

def _stats(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Aggregate a 2-D feature matrix (n_features × n_frames) into
    a 1-D vector of [mean, std, min, max] per feature.

    Parameters
    ----------
    feature_matrix : np.ndarray  shape (n_features, n_frames)

    Returns
    -------
    np.ndarray  shape (n_features × 4,)
    """
    mean = np.mean(feature_matrix, axis=1)
    std  = np.std(feature_matrix, axis=1)
    _min = np.min(feature_matrix, axis=1)
    _max = np.max(feature_matrix, axis=1)
    return np.concatenate([mean, std, _min, _max])


# ─────────────────────────────────────────────
#  Individual Feature Extractors
# ─────────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC + Δ-MFCC + ΔΔ-MFCC and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (n_mfcc × 4 × 3,)
    """
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([_stats(mfcc), _stats(delta), _stats(delta2)])


def extract_chroma(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Chroma STFT (12 pitch classes) and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (12 × 4,)
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return _stats(chroma)


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                             n_mels: int = N_MELS) -> np.ndarray:
    """
    Extract Mel Spectrogram (log scale) and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (n_mels × 4,)
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH,
                                          fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _stats(log_mel)


def extract_spectral_contrast(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Spectral Contrast (7 frequency bands) and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (7 × 4,)
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                   n_fft=N_FFT,
                                                   hop_length=HOP_LENGTH)
    return _stats(contrast)


def extract_zcr(y: np.ndarray) -> np.ndarray:
    """
    Extract Zero-Crossing Rate and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (4,)
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return _stats(zcr)


def extract_rms(y: np.ndarray) -> np.ndarray:
    """
    Extract RMS Energy and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (4,)
    """
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return _stats(rms)


def extract_spectral_rolloff(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Spectral Rolloff and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (4,)
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                hop_length=HOP_LENGTH)
    return _stats(rolloff)


def extract_tonnetz(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Tonnetz (tonal centroid features) and aggregate statistics.

    Returns
    -------
    np.ndarray  shape (6 × 4,)
    """
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    return _stats(tonnetz)


# ─────────────────────────────────────────────
#  Master Feature Extractor
# ─────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE,
                     config: dict = None) -> np.ndarray:
    """
    Extract all enabled features and return a single flat feature vector.

    Parameters
    ----------
    y : np.ndarray
        Preprocessed audio signal.
    sr : int
        Sample rate.
    config : dict
        Feature config flags (defaults to FEATURE_CONFIG from config.py).

    Returns
    -------
    np.ndarray : 1-D feature vector.
    """
    if config is None:
        config = FEATURE_CONFIG

    parts = []

    if config.get("mfcc", True):
        parts.append(extract_mfcc(y, sr))

    if config.get("chroma", True):
        parts.append(extract_chroma(y, sr))

    if config.get("mel_spectrogram", True):
        parts.append(extract_mel_spectrogram(y, sr))

    if config.get("spectral_contrast", True):
        parts.append(extract_spectral_contrast(y, sr))

    if config.get("zero_crossing_rate", True):
        parts.append(extract_zcr(y))

    if config.get("rms_energy", True):
        parts.append(extract_rms(y))

    if config.get("spectral_rolloff", True):
        parts.append(extract_spectral_rolloff(y, sr))

    if config.get("tonnetz", True):
        parts.append(extract_tonnetz(y, sr))

    feature_vector = np.concatenate(parts)
    return feature_vector.astype(np.float32)


def get_feature_names(config: dict = None) -> list[str]:
    """
    Generate human-readable names for each feature dimension,
    in the same order as extract_features().

    Returns
    -------
    list[str] : Feature names.
    """
    if config is None:
        config = FEATURE_CONFIG

    names = []
    stats_labels = ["mean", "std", "min", "max"]

    def _add(prefix, n):
        for i in range(n):
            for s in stats_labels:
                names.append(f"{prefix}_{i}_{s}")

    if config.get("mfcc", True):
        _add("mfcc", N_MFCC)
        _add("mfcc_delta", N_MFCC)
        _add("mfcc_delta2", N_MFCC)

    if config.get("chroma", True):
        _add("chroma", 12)

    if config.get("mel_spectrogram", True):
        _add("mel", N_MELS)

    if config.get("spectral_contrast", True):
        _add("spectral_contrast", 7)

    if config.get("zero_crossing_rate", True):
        for s in stats_labels:
            names.append(f"zcr_{s}")

    if config.get("rms_energy", True):
        for s in stats_labels:
            names.append(f"rms_{s}")

    if config.get("spectral_rolloff", True):
        for s in stats_labels:
            names.append(f"rolloff_{s}")

    if config.get("tonnetz", True):
        _add("tonnetz", 6)

    return names


def save_feature_names(names: list[str]) -> None:
    """Persist feature names to JSON for inference-time validation."""
    with open(FEATURE_NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)
    logger.info(f"Saved {len(names)} feature names to {FEATURE_NAMES_FILE}")


def load_feature_names() -> list[str]:
    """Load previously saved feature names."""
    with open(FEATURE_NAMES_FILE, "r") as f:
        return json.load(f)
