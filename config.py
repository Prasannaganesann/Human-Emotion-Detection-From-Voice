"""
config.py
=========
Central configuration for the Human Emotion Detection system.
All tunable parameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Project Paths
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = BASE_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = BASE_DIR / "saved_models"
LOGS_DIR        = BASE_DIR / "logs"
DB_PATH         = BASE_DIR / "data" / "session_history.db"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  Audio Settings
# ─────────────────────────────────────────────
SAMPLE_RATE        = 22050          # Hz – standard for speech
DURATION           = 3.0            # seconds per clip (padded/trimmed)
HOP_LENGTH         = 512
N_FFT              = 2048
N_MELS             = 128
N_MFCC             = 40
FMAX               = 8000           # Hz
RECORDING_DURATION = 5              # seconds for live recording

# ─────────────────────────────────────────────
#  Emotion Classes (RAVDESS mapping)
# ─────────────────────────────────────────────
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Subset we train on (you can expand)
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]

EMOTION_COLORS = {
    "neutral":  "#94A3B8",
    "happy":    "#FCD34D",
    "sad":      "#60A5FA",
    "angry":    "#F87171",
    "fearful":  "#A78BFA",
    "calm":     "#6EE7B7",
    "disgust":  "#FB923C",
    "surprised":"#F472B6",
}

EMOTION_EMOJIS = {
    "neutral":  "😐",
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "fearful":  "😨",
    "calm":     "😌",
    "disgust":  "🤢",
    "surprised":"😲",
}

# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────
FEATURE_CONFIG = {
    "mfcc":             True,
    "mfcc_delta":       True,
    "mfcc_delta2":      True,
    "chroma":           True,
    "mel_spectrogram":  True,
    "spectral_contrast":True,
    "zero_crossing_rate":True,
    "rms_energy":       True,
    "spectral_rolloff": True,
    "tonnetz":          True,
}

# ─────────────────────────────────────────────
#  Model Settings
# ─────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5

# Best model filename
BEST_MODEL_FILE = MODELS_DIR / "best_emotion_model.pkl"
SCALER_FILE     = MODELS_DIR / "feature_scaler.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"

# ─────────────────────────────────────────────
#  API Settings
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
