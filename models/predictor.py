"""
predictor.py
============
Inference engine that wraps the trained model, scaler, and label encoder.
Provides a clean API consumed by both the Streamlit app and the FastAPI backend.

Key design change from v1: removed the __new__ singleton pattern so that
model hot-swapping actually works when the user selects a different model
in the sidebar.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BEST_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE,
    EMOTION_EMOJIS, EMOTION_COLORS, MODELS_DIR
)
from utils.preprocessing import preprocess_audio, get_audio_info
from utils.feature_extraction import extract_features

logger = logging.getLogger(__name__)

# Confidence threshold below which the prediction is marked "uncertain"
UNCERTAINTY_THRESHOLD = 0.35


class EmotionPredictor:
    """
    Wrapper that loads a model + scaler + label encoder and exposes
    a simple ``predict()`` method.

    Unlike the old singleton, this class can be freely instantiated
    multiple times so Streamlit's ``st.cache_resource`` can cache one
    instance *per model path*.

    Usage
    -----
    predictor = EmotionPredictor()
    predictor.load("saved_models/svm_rbf_model.pkl")
    result    = predictor.predict_from_file("audio.wav")
    """

    def __init__(self):
        self._loaded = False
        self.model = None
        self.scaler = None
        self.le = None
        self.classes = []
        self.model_name = ""

    # ─────────────────────────────────────────
    #  Loading
    # ─────────────────────────────────────────

    def load(self, model_path: str = None) -> None:
        """Load model, scaler, and label encoder from disk."""
        model_file = Path(model_path) if model_path else BEST_MODEL_FILE

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found at {model_file}. "
                "Run generate_demo_model.py or training/train_model.py first."
            )
        if not SCALER_FILE.exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_FILE}. "
                "Run generate_demo_model.py or training/train_model.py first."
            )
        if not LABEL_ENCODER_FILE.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {LABEL_ENCODER_FILE}. "
                "Run generate_demo_model.py or training/train_model.py first."
            )

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(SCALER_FILE)
        self.le = joblib.load(LABEL_ENCODER_FILE)
        self.classes = self.le.classes_.tolist()
        self.model_name = model_file.stem
        self._loaded = True
        logger.info(f"Loaded model: {self.model_name}  |  classes: {self.classes}")

    # ─────────────────────────────────────────
    #  Core Prediction
    # ─────────────────────────────────────────

    def predict(self, y: np.ndarray, sr: int,
                audio_info: dict = None) -> dict:
        """
        Run inference on a preprocessed audio array.

        Parameters
        ----------
        y : np.ndarray
            Preprocessed audio signal.
        sr : int
            Sample rate.
        audio_info : dict, optional
            Metadata dict (from get_audio_info).

        Returns
        -------
        dict : Full prediction result with probabilities, emoji, color, etc.
        """
        if not self._loaded:
            self.load()

        feat_vec = extract_features(y, sr)
        feat_scaled = self.scaler.transform(feat_vec.reshape(1, -1))

        pred_idx = self.model.predict(feat_scaled)[0]
        proba_arr = self.model.predict_proba(feat_scaled)[0]

        emotion = self.le.inverse_transform([pred_idx])[0]
        confidence = float(proba_arr[pred_idx])
        probs = {cls: float(p) for cls, p in zip(self.classes, proba_arr)}

        # Mark as uncertain when confidence is below threshold
        is_uncertain = confidence < UNCERTAINTY_THRESHOLD
        display_emotion = "uncertain" if is_uncertain else emotion

        return {
            "emotion":        display_emotion,
            "raw_emotion":    emotion,
            "confidence":     round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "is_uncertain":   is_uncertain,
            "emoji":          "❓" if is_uncertain else EMOTION_EMOJIS.get(emotion, "🎙️"),
            "color":          "#6B7280" if is_uncertain else EMOTION_COLORS.get(emotion, "#94A3B8"),
            "probabilities":  probs,
            "model_name":     self.model_name,
            "audio_info":     audio_info or {},
        }

    # ─────────────────────────────────────────
    #  Convenience Wrappers
    # ─────────────────────────────────────────

    def predict_from_file(self, file_path: str) -> dict:
        """Preprocess + predict from a file path."""
        y, sr = preprocess_audio(file_path=file_path)
        info = get_audio_info(y, sr)
        return self.predict(y, sr, audio_info=info)

    def predict_from_bytes(self, audio_bytes: bytes,
                           filename: str = "upload") -> dict:
        """Preprocess + predict from raw bytes."""
        y, sr = preprocess_audio(audio_bytes=audio_bytes)
        info = get_audio_info(y, sr)
        info["filename"] = filename
        return self.predict(y, sr, audio_info=info)

    def predict_from_array(self, y: np.ndarray, sr: int) -> dict:
        """Preprocess + predict from an already-loaded array."""
        from utils.preprocessing import (
            trim_silence, normalize_audio, pad_or_trim, reduce_noise
        )
        y = trim_silence(y)
        y = normalize_audio(y)
        y = pad_or_trim(y, sr)
        info = get_audio_info(y, sr)
        return self.predict(y, sr, audio_info=info)

    def predict_batch(self, file_paths: list[str]) -> list[dict]:
        """Predict emotions for a batch of audio files."""
        results = []
        for fp in file_paths:
            try:
                results.append(self.predict_from_file(fp))
            except Exception as e:
                results.append({"file": fp, "error": str(e)})
        return results

    def predict_batch_bytes(self, items: list[tuple[bytes, str]]) -> list[dict]:
        """
        Predict emotions for a batch of (audio_bytes, filename) tuples.
        Returns a list of result dicts, one per input.
        """
        results = []
        for audio_bytes, filename in items:
            try:
                results.append(self.predict_from_bytes(audio_bytes, filename))
            except Exception as e:
                results.append({"file": filename, "error": str(e)})
        return results

    # ─────────────────────────────────────────
    #  Available Models
    # ─────────────────────────────────────────

    @staticmethod
    def list_available_models() -> list[str]:
        """Return all saved model pkl files."""
        return [str(p) for p in MODELS_DIR.glob("*_model.pkl")]
