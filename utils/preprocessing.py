"""
preprocessing.py
================
Audio loading, denoising, normalization, and segmentation utilities.
All functions return numpy arrays ready for feature extraction.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import central config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, DURATION, HOP_LENGTH, N_FFT


# ─────────────────────────────────────────────
#  Core Loader
# ─────────────────────────────────────────────

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Parameters
    ----------
    file_path : str
        Path to audio file (.wav, .mp3, .flac, etc.)
    sr : int
        Target sample rate (default: 22050)

    Returns
    -------
    (y, sr) : tuple
        Audio time series and sample rate.
    """
    try:
        y, original_sr = librosa.load(file_path, sr=sr, mono=True)
        logger.debug(f"Loaded {file_path} | sr={sr} | samples={len(y)}")
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise


def load_audio_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes (e.g., from Streamlit uploader or API).
    Falls back to librosa for MP3/M4A which soundfile cannot decode.
    """
    import io
    buf = io.BytesIO(audio_bytes)
    try:
        y, original_sr = sf.read(buf)
    except Exception:
        # soundfile cannot read MP3/M4A — fall back to librosa (requires ffmpeg)
        buf.seek(0)
        try:
            y, original_sr = librosa.load(buf, sr=sr, mono=True)
            return y.astype(np.float32), sr
        except Exception as e:
            raise ValueError(
                f"Cannot decode audio. Supported formats: WAV, FLAC, OGG. "
                f"For MP3/M4A, install ffmpeg. Error: {e}"
            ) from e
    if y.ndim > 1:
        y = y.mean(axis=1)             # Stereo → mono
    y = librosa.resample(y.astype(np.float32), orig_sr=original_sr, target_sr=sr)
    return y, sr


# ─────────────────────────────────────────────
#  Preprocessing Pipeline
# ─────────────────────────────────────────────

def trim_silence(y: np.ndarray, top_db: int = 25) -> np.ndarray:
    """
    Trim leading/trailing silence using librosa's top_db threshold.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    top_db : int
        Threshold in dB below reference (default 25 dB).

    Returns
    -------
    np.ndarray : Trimmed audio signal.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1] range.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.

    Returns
    -------
    np.ndarray : Normalized audio signal.
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


def pad_or_trim(y: np.ndarray, sr: int = SAMPLE_RATE,
                duration: float = DURATION) -> np.ndarray:
    """
    Pad or trim audio to a fixed duration.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    duration : float
        Target duration in seconds.

    Returns
    -------
    np.ndarray : Fixed-length audio signal.
    """
    target_length = int(sr * duration)
    if len(y) < target_length:
        # Pad with zeros
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        # Trim to target length
        y = y[:target_length]
    return y


def reduce_noise(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Simple spectral noise reduction using a noise profile estimate.
    Uses the first 0.3s as a noise reference frame.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray : Denoised audio signal.
    """
    # Estimate noise profile from the first 300ms
    noise_sample_len = min(int(0.3 * sr), len(y) // 4)
    noise_ref = y[:noise_sample_len]

    # STFT on full signal
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_noise = librosa.stft(noise_ref, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Noise magnitude profile (mean across time)
    noise_mag = np.mean(np.abs(D_noise), axis=1, keepdims=True)

    # Spectral subtraction
    mag, phase = np.abs(D), np.angle(D)
    mag_denoised = np.maximum(mag - noise_mag, 0.0)

    # Reconstruct signal
    D_denoised = mag_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(D_denoised, hop_length=HOP_LENGTH)

    # Match original length
    y_denoised = pad_or_trim(y_denoised, sr, duration=len(y) / sr)
    return y_denoised


def preprocess_audio(file_path: str = None,
                     audio_bytes: bytes = None,
                     audio_array: np.ndarray = None,
                     sr: int = SAMPLE_RATE,
                     apply_noise_reduction: bool = True) -> tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline:
      1. Load audio
      2. Trim silence
      3. Noise reduction (optional)
      4. Peak normalization
      5. Pad/trim to fixed duration

    Parameters
    ----------
    file_path : str, optional
        Path to audio file.
    audio_bytes : bytes, optional
        Raw audio bytes (Streamlit upload / API).
    audio_array : np.ndarray, optional
        Pre-loaded audio array.
    sr : int
        Target sample rate.
    apply_noise_reduction : bool
        Whether to apply spectral noise reduction.

    Returns
    -------
    (y, sr) : tuple
        Preprocessed audio signal and sample rate.
    """
    # Step 1: Load
    if file_path is not None:
        y, sr = load_audio(file_path, sr)
    elif audio_bytes is not None:
        y, sr = load_audio_bytes(audio_bytes, sr)
    elif audio_array is not None:
        y = audio_array.copy()
    else:
        raise ValueError("Provide one of: file_path, audio_bytes, or audio_array")

    # Step 2: Trim silence
    y = trim_silence(y)

    # Step 3: Noise reduction
    if apply_noise_reduction and len(y) > int(0.5 * sr):
        try:
            y = reduce_noise(y, sr)
        except Exception as ex:
            logger.warning(f"Noise reduction skipped: {ex}")

    # Step 4: Normalize
    y = normalize_audio(y)

    # Step 5: Fixed length
    y = pad_or_trim(y, sr)

    return y, sr


def get_audio_info(y: np.ndarray, sr: int) -> dict:
    """
    Compute basic statistics about an audio clip.

    Returns
    -------
    dict with duration, rms, max_amplitude, etc.
    """
    return {
        "duration_s":    round(len(y) / sr, 3),
        "sample_rate":   sr,
        "samples":       len(y),
        "rms":           round(float(np.sqrt(np.mean(y**2))), 6),
        "max_amplitude": round(float(np.max(np.abs(y))), 6),
        "min_amplitude": round(float(np.min(np.abs(y))), 6),
    }
