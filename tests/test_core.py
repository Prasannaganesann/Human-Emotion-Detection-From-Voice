"""
tests/test_core.py
==================
Lightweight test suite covering preprocessing, feature extraction, and
predictor loading. Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import SAMPLE_RATE, DURATION
from utils.preprocessing import (
    trim_silence, normalize_audio, pad_or_trim,
    reduce_noise, preprocess_audio, get_audio_info,
)
from utils.feature_extraction import extract_features, get_feature_names


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sine_wave():
    """440 Hz sine wave, 2 seconds."""
    sr = SAMPLE_RATE
    t = np.linspace(0, 2.0, int(sr * 2), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


@pytest.fixture
def silent_padded():
    """Audio with leading/trailing silence."""
    sr = SAMPLE_RATE
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    y = np.concatenate([silence, tone, silence])
    return y, sr


# ─────────────────────────────────────────────
#  Preprocessing Tests
# ─────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_peaks_at_one(self, sine_wave):
        y, _ = sine_wave
        y_norm = normalize_audio(y)
        assert abs(np.max(np.abs(y_norm)) - 1.0) < 1e-6

    def test_normalize_silent_signal(self):
        y = np.zeros(1000, dtype=np.float32)
        y_norm = normalize_audio(y)
        assert np.allclose(y_norm, 0.0)

    def test_pad_short_audio(self, sine_wave):
        y, sr = sine_wave
        target = int(sr * DURATION)
        y_fixed = pad_or_trim(y, sr, DURATION)
        assert len(y_fixed) == target

    def test_trim_long_audio(self):
        sr = SAMPLE_RATE
        y = np.random.randn(sr * 10).astype(np.float32)
        y_fixed = pad_or_trim(y, sr, DURATION)
        assert len(y_fixed) == int(sr * DURATION)

    def test_trim_silence_reduces_length(self, silent_padded):
        y, _ = silent_padded
        y_trimmed = trim_silence(y)
        assert len(y_trimmed) < len(y)

    def test_reduce_noise_same_length(self, sine_wave):
        y, sr = sine_wave
        y_dn = reduce_noise(y, sr)
        assert len(y_dn) == len(y)

    def test_get_audio_info_keys(self, sine_wave):
        y, sr = sine_wave
        info = get_audio_info(y, sr)
        assert "duration_s" in info
        assert "sample_rate" in info
        assert "rms" in info
        assert info["sample_rate"] == sr

    def test_preprocess_from_array(self, sine_wave):
        y, sr = sine_wave
        y_out, sr_out = preprocess_audio(audio_array=y, sr=sr)
        assert sr_out == sr
        assert len(y_out) == int(sr * DURATION)


# ─────────────────────────────────────────────
#  Feature Extraction Tests
# ─────────────────────────────────────────────

class TestFeatureExtraction:
    def test_feature_vector_is_1d(self, sine_wave):
        y, sr = sine_wave
        y = pad_or_trim(y, sr, DURATION)
        feat = extract_features(y, sr)
        assert feat.ndim == 1

    def test_feature_vector_length_matches_names(self, sine_wave):
        y, sr = sine_wave
        y = pad_or_trim(y, sr, DURATION)
        feat = extract_features(y, sr)
        names = get_feature_names()
        assert len(feat) == len(names), (
            f"Feature vector ({len(feat)}) != names ({len(names)})"
        )

    def test_features_are_finite(self, sine_wave):
        y, sr = sine_wave
        y = pad_or_trim(y, sr, DURATION)
        feat = extract_features(y, sr)
        assert np.all(np.isfinite(feat))

    def test_feature_dtype_is_float32(self, sine_wave):
        y, sr = sine_wave
        y = pad_or_trim(y, sr, DURATION)
        feat = extract_features(y, sr)
        assert feat.dtype == np.float32


# ─────────────────────────────────────────────
#  Predictor Tests
# ─────────────────────────────────────────────

class TestPredictor:
    def test_predictor_load_fails_on_missing_model(self):
        from models.predictor import EmotionPredictor
        p = EmotionPredictor()
        with pytest.raises(FileNotFoundError):
            p.load("nonexistent_model.pkl")

    def test_predictor_list_models_returns_list(self):
        from models.predictor import EmotionPredictor
        models = EmotionPredictor.list_available_models()
        assert isinstance(models, list)

    def test_predictor_load_and_predict(self, sine_wave):
        """Only runs if a demo model has been generated."""
        from models.predictor import EmotionPredictor
        from config import BEST_MODEL_FILE
        if not BEST_MODEL_FILE.exists():
            pytest.skip("No model file — run generate_demo_model.py first")

        p = EmotionPredictor()
        p.load()
        y, sr = sine_wave
        y = pad_or_trim(y, sr, DURATION)
        result = p.predict(y, sr)
        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0 <= result["confidence"] <= 1
